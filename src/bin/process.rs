/// A small program that lets us process
/// a directory of Dora's images
//
///
/// Author: Benjamin Blundell
/// Email: me@benjamin.computer

extern crate rand;
extern crate image;
extern crate nalgebra as na;
extern crate probability;
extern crate scoped_threadpool;
extern crate ndarray;
extern crate gtk;
extern crate gio;
extern crate gdk_pixbuf;
extern crate glib;
extern crate tiff;
extern crate fitrs;

use gtk::prelude::*;
use gio::prelude::*;
use gdk_pixbuf::Pixbuf;
use gdk_pixbuf::Colorspace;
use glib::Bytes;
use glib::clone;

use std::env;
use std::fmt;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use rand::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{cell::Cell, rc::Rc, cell::RefCell};
use tiff::decoder::{ifd, Decoder, DecodingResult};
use tiff::ColorType;
use std::process;
use fitrs::{Fits, Hdu};
use rand::distributions::Uniform;
use scoped_threadpool::Pool;
use std::sync::mpsc::channel;
use pbr::ProgressBar;
use ndarray::{Slice, SliceInfo, s, Array1};
use gtk::{Application, ApplicationWindow, Button};
use dora_explorer::dora_tiff::save_fits;
use dora_explorer::dora_tiff::tiff_to_vec;
use dora_explorer::dora_tiff::tiff_to_stack;
use dora_explorer::dora_tiff::save_tiff_stack;
use dora_explorer::dora_tiff::get_size;
use dora_explorer::dora_tiff::aug_vec;
use dora_explorer::dora_tiff::aug_stack;
use dora_explorer::dora_tiff::gauss_blur;
use dora_explorer::dora_tiff::Direction;

// render function. Breaks up the list of paths into chunks for each thread
fn render (image_paths : &Vec<PathBuf>, out_path : &String,  nthreads : u32, gauss : f32) {
    // Split into threads here I think
    let pi = std::f32::consts::PI;
    let (tx, rx) = channel();
    let mut progress : i32 = 0;
    let mut pool = Pool::new(nthreads);

    let num_runs = image_paths.len() as u32;
    let truns = (num_runs / nthreads) as u32;
    let spare = (num_runs % nthreads) as u32;
    let mut pb = ProgressBar::new(num_runs as u64);
    pb.format("╢▌▌░╟");

    pool.scoped(|scoped| {
        for _t in 0..nthreads {
            let tx = tx.clone();
            let start : usize = (_t * truns) as usize;
            let mut end = ((_t + 1)  * truns) as usize;
            if _t == nthreads - 1 { end = end + (spare as usize); }
            let cslice = &image_paths[start..end];
           
            scoped.execute( move || { 
                let mut rng = thread_rng();
                let side = Uniform::new(-pi, pi);

                for _i in 0..cslice.len() {
                    let (width, height) = get_size(&cslice[_i]);
                    let (timg, minp, maxp, levels, _, _) = tiff_to_vec(&cslice[_i]);
                    let fidx = format!("/image_{:06}.fits", ((start + _i) * 4) as usize);
                    let mut fitspath = out_path.clone();
                    fitspath.push_str(&fidx);
                    save_fits(&timg, &fitspath, width, height);

                    let (img_stack, _, _, _) = tiff_to_stack(&cslice[_i]);

                    let tidx = format!("/image_{:06}.tiff", ((start + _i) * 4) as usize);
                    let mut tiffpath = out_path.clone();
                    tiffpath.push_str(&tidx);
                    save_tiff_stack(&img_stack, &tiffpath, width, height);

                    // Perform a gauss blur
                    let gimg = gauss_blur(&timg, gauss);

                    // Now Augment
                    let fidx1 = format!("/image_{:06}.fits", ((start + _i) * 4 + 1) as usize);
                    fitspath = out_path.clone();
                    fitspath.push_str(&fidx1);
                    let aimg1 = aug_vec(&gimg, Direction::Right);
                    save_fits(&aimg1, &fitspath, width, height);

                    let tidx1 = format!("/image_{:06}.tiff", ((start + _i) * 4 + 1) as usize);
                    let mut tiffpath = out_path.clone();
                    tiffpath.push_str(&tidx1);
                    let astack1 = aug_stack(&img_stack, Direction::Right);
                    save_tiff_stack(&astack1, &tiffpath, width, height);

                    let fidx2 = format!("/image_{:06}.fits", ((start + _i) * 4 + 2) as usize);
                    fitspath = out_path.clone();
                    fitspath.push_str(&fidx2);
                    let aimg2 = aug_vec(&gimg, Direction::Down);
                    save_fits(&aimg2, &fitspath, width, height);

                    let tidx2 = format!("/image_{:06}.tiff", ((start + _i) * 4 + 2) as usize);
                    let mut tiffpath = out_path.clone();
                    tiffpath.push_str(&tidx2);
                    let astack2 = aug_stack(&img_stack, Direction::Down);
                    save_tiff_stack(&astack2, &tiffpath, width, height);

                    let fidx3 = format!("/image_{:06}.fits", ((start + _i) * 4 + 3) as usize);
                    fitspath = out_path.clone();
                    fitspath.push_str(&fidx3);
                    let aimg3 = aug_vec(&gimg, Direction::Left);
                    save_fits(&aimg3, &fitspath, width, height);

                    let tidx3 = format!("/image_{:06}.tiff", ((start + _i) * 4 + 3) as usize);
                    let mut tiffpath = out_path.clone();
                    tiffpath.push_str(&tidx3);
                    let astack3 = aug_stack(&img_stack, Direction::Left);
                    save_tiff_stack(&astack3, &tiffpath, width, height);
            
                    tx.send(_i).unwrap();
                }
            });
        }

        // Update our progress bar
        while progress < num_runs as i32 {
            match rx.try_recv() {
                Ok(_a) => {
                    pb.inc();
                    progress = progress + 1;
                }, Err(_e) => {}
            }
        }
    });
}

// Main entry point where we parse the command line, read in all the images
// into a path buffer and set off all the threads.

fn main() {
    let args: Vec<_> = env::args().collect();
    let mut image_files : Vec<PathBuf> = vec!();
    
    if args.len() < 4 {
        println!("Usage: explorer <path to directory of tiff files>
            <output dir> <num threads> <gauss> <OPTIONAL filter>"); 
        process::exit(1);
    }

    let mut filter : String = String::new();
    if args.len() == 6 {
        filter.push_str(&args[5]);
    }

    let paths = fs::read_dir(Path::new(&args[1])).unwrap();
    let nthreads = args[3].parse::<u32>().unwrap();
    let gauss = args[4].parse::<f32>().unwrap();
    let mut idf = 0;

    for path in paths {
        match path {
            Ok(file) => {
                let filename = file.file_name();
                let tx = filename.to_str().unwrap();
                let mut accept : bool = false;
                // go for deconvolved(?) and tifs
               
                if tx.contains("tif") && tx.contains(filter.as_str()) {
                    accept = true;
                }
                
                if accept {
                    println!("Found tiff: {}", tx);
                    let mut owned_string: String = args[1].to_owned();
                    let borrowed_string: &str = "/";
                    owned_string.push_str(borrowed_string);
                    owned_string.push_str(&tx.to_string());
                    image_files.push(PathBuf::from(owned_string));
                }
            },
            Err(e) => {
                println!("Error walking directory.");
            }
        }
        idf = idf + 1;
       
    }

    render(&image_files, &args[2], nthreads, gauss);
}