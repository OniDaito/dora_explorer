/// A small program that lets us view
/// a directory of Dora's images
///
/// Using a little gtk-rs
/// https://gtk-rs.org/docs-src/tutorial/
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

static WIDTH : u32 = 128;
static HEIGHT : u32 = 128;
static SHRINK : f32 = 0.95;


// Holds our models and our GTK+ application
pub struct Explorer {
    app: gtk::Application,
    image_paths : Vec<PathBuf>,
    image_index : Cell<usize>, // use this so we can mutate it later
    accept_count : Cell<usize>,
    output_path : PathBuf,
    image_buffer : RefCell<Vec<Vec<f32>>>
}

// Convert our model into a gtk::Image that we can present to
// the screen.

fn get_image(path : &Path) -> (gtk::Image, Vec<Vec<f32>>) {
    let img_file = File::open(path).expect("Cannot find test image!");
    let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

    assert_eq!(decoder.colortype().unwrap(), ColorType::Gray(16));
    let img_res = decoder.read_image().unwrap();

    // Check the image size here

    // Our buffer - we sum all the image here and then scale
    let mut img_buffer : Vec<Vec<f32>> = vec![];
    for y in 0..HEIGHT {
        let mut row  : Vec<f32> = vec![];
        for x in 0..WIDTH {
            row.push(0 as f32);
        }
        img_buffer.push(row);
    }

    // Final buffer that we use that is a little smaller - u8
    // and not u16, but also RGB, just to make GTK happy.
    let mut final_buffer : Vec<u8> = vec![];
    for y in 0..HEIGHT {
        let mut row  : Vec<u8> = vec![];
        for x in 0..WIDTH {
            // GTK insists we have RGB so we triple everything :/
            for _ in 0..3 {
                final_buffer.push(0 as u8);
            }
        }
    }
   
    // Now we've decoded, lets update the img_buffer
    if let DecodingResult::U16(img_res) = img_res {
        let mut levels : usize = 0;
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                img_buffer[y][x] = (img_res[y * (HEIGHT as usize) + x] as f32);
            }
        }

        while decoder.more_images() {
            let next_res = decoder.next_image();
            match next_res {
                Ok(res) => {   
                    let img_next = decoder.read_image().unwrap();
                    if let DecodingResult::U16(img_next) = img_next {
                        levels += 1;
                        for y in 0..HEIGHT as usize {
                            for x in 0..WIDTH as usize {
                                img_buffer[y][x] += (img_next[y * (HEIGHT as usize) + x] as f32);
                            }
                        } 
                    }
                },
                Err(_) => {}
            }
        }
        // We take an average rather than a total sum
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                img_buffer[y][x] = img_buffer[y][x] / (levels as f32);
            }
        }

        // Find min/max
        let mut minp : f32 = 1e12; // we might end up overflowing!
        let mut maxp : f32 = 0.0;
        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                if (img_buffer[y][x] as f32) > maxp { maxp = img_buffer[y][x] as f32; }
                if (img_buffer[y][x] as f32) < minp { minp = img_buffer[y][x] as f32; }
            }
        }

        for y in 0..HEIGHT as usize {
            for x in 0..WIDTH as usize {
                let colour = (img_buffer[y][x] / maxp * 255.0) as u8;
                let idx = (y * (HEIGHT as usize) + x) * 3;
                final_buffer[idx] = colour;
                final_buffer[idx+1] = colour;
                final_buffer[idx+2] = colour;
            }
        } 

        let b = Bytes::from(&final_buffer);

        println!("Succesfully read {} which has {} levels.", path.display(), levels);

        // Convert down the tiff so we can see it.
        
        let pixybuf = Pixbuf::new_from_bytes(&b,
            Colorspace::Rgb,
            false, 
            8,
            WIDTH as i32,
            HEIGHT as i32,
            (WIDTH * 3 * 1) as i32
        );

        let image : gtk::Image = gtk::Image::new_from_pixbuf(Some(&pixybuf));
        return (image, img_buffer);

    } else {
        panic!("Wrong data type");
    }

    let image: gtk::Image = gtk::Image::new();
    (image, img_buffer)
}

// Save out the fits image
pub fn save_fits(img : &Vec<Vec<f32>>, filename : &String) {
    let mut data : Vec<f32> = (0..HEIGHT)
        .map(|i| (0..WIDTH).map(
               move |j| (i + j) as f32)).flatten().collect();

    for _y in 0..HEIGHT {
        for _x in 0..WIDTH {
            let idx : usize = (_y * WIDTH +_x ) as usize; 
            data[idx] = img[_x as usize][(HEIGHT - _y - 1) as usize];
            // / intensity * MULTFAC;
        }
    }

    let mut primary_hdu = 
        Hdu::new(&[WIDTH as usize , HEIGHT as usize], data);
    // Insert values in header
    primary_hdu.insert("NORMALISATION", "NONE");
    primary_hdu.insert("WIDTH", WIDTH as i32);
    primary_hdu.insert("HEIGHT", HEIGHT as i32);
    Fits::create(filename, primary_hdu).expect("Failed to create");  
}

pub fn copy_buffer(in_buff : &Vec<Vec<f32>>, out_buff : &mut Vec<Vec<f32>>) {
    for _y in 0..HEIGHT as usize {
        for _x in 0..WIDTH as usize {
            out_buff[_y][_x] = in_buff[_y][_x];
        }
    }
}


// Our chooser struct/class implementation. Mostly just runs the GTK
// and keeps a hold on our models.
impl Explorer {
    pub fn new(image_paths : Vec<PathBuf>, output_path : PathBuf) -> Rc<Self> {
        let app = Application::new(
            Some("com.github.gtk-rs.examples.basic"),
            Default::default(),
        ).expect("failed to initialize GTK application");

        let mut image_index : Cell<usize> = Cell::new(0);
        let mut accept_count : Cell<usize> = Cell::new(0);

        let mut tbuf : Vec<Vec<f32>> = vec![];
        for y in 0..HEIGHT {
            let mut row  : Vec<f32> = vec![];
            for x in 0..WIDTH {
                row.push(0 as f32);
            }
            tbuf.push(row);
        }

        let mut image_buffer : RefCell<Vec<Vec<f32>>> = RefCell::new(tbuf);

        let explorer = Rc::new(Self {
            app,
            image_paths,
            image_index,
            accept_count,
            output_path,
            image_buffer
        });

        explorer
    }

    pub fn run(&self, app: Rc<Self>) {
        let app = app.clone();
        let args: Vec<String> = env::args().collect();
 
        self.app.connect_activate( move |gtkapp| {
            let window = ApplicationWindow::new(gtkapp);
            window.set_title("Dora Explorer");
            window.set_default_size(350, 350);
            let vbox = gtk::Box::new(gtk::Orientation::Vertical, 3);
            let ibox = gtk::Box::new(gtk::Orientation::Horizontal, 1);
            let hbox = gtk::Box::new(gtk::Orientation::Horizontal, 3);
            let (image, buffer) = get_image(&(app.image_paths[0]));
            copy_buffer(&buffer, &mut app.image_buffer.borrow_mut());

            ibox.add(&image);
            vbox.add(&ibox);
            vbox.add(&hbox);
            window.add(&vbox);

            // Now look at buttons
            let button_accept = Button::new_with_label("Accept");
            let ibox_arc = Arc::new(Mutex::new(ibox));
            let ibox_accept = ibox_arc.clone();
            let mut app_accept = app.clone();

            let mut i : i32 = 0;
            let button_click = || { i + 1 };
            
            // Accept button
            button_accept.connect_clicked( move |button| {
                println!("Accepted {}", app_accept.image_index.get());
                let mi = app_accept.image_index.get();
                if mi + 1 >= app_accept.image_paths.len() {
                    println!("All images checked!");
                    return;
                }

                // Get the current filename and save out the buffer
                let fidx = format!("/image_{:06}.fits", app_accept.accept_count.get() as usize);
                let mut fitspath = String::from(app_accept.output_path.to_str().unwrap());
                fitspath.push_str(&fidx);
                save_fits(&app_accept.image_buffer.borrow_mut(), &fitspath);

                // Increment accept count
                let ai = app_accept.accept_count.get();
                app_accept.accept_count.set(ai + 1);

                // Now move on to the next image
                let ibox_ref = ibox_accept.lock().unwrap();
                let children : Vec<gtk::Widget> = (*ibox_ref).get_children();
                app_accept.image_index.set(mi + 1);
                let (image, buffer) = get_image(&(app_accept.image_paths[mi + 1]));
                copy_buffer(&buffer, &mut app_accept.image_buffer.borrow_mut());

                (*ibox_ref).remove(&children[0]);
                (*ibox_ref).add(&image);
                let window_ref = (*ibox_ref).get_parent().unwrap();
                window_ref.show_all();
            });

            hbox.add(&button_accept);

            let button_reject = Button::new_with_label("Reject");
            let ibox_reject = ibox_arc.clone();
            let mut app_reject = app.clone();

            button_reject.connect_clicked( move |button| {
                println!("Rejected {}", app_reject.image_index.get());

                // Check we aren't overrunning
                let mi = app_reject.image_index.get();
                if mi + 1 >= app_reject.image_paths.len() {
                    println!("All images checked!");
                    process::exit(1);
                }

                let ibox_ref = ibox_reject.lock().unwrap();
                let children : Vec<gtk::Widget> = (*ibox_ref).get_children();
                app_reject.image_index.set(mi + 1);
                let (image, buffer) = get_image(&(app_reject.image_paths[mi + 1]));
                copy_buffer(&buffer, &mut app_reject.image_buffer.borrow_mut());
                (*ibox_ref).remove(&children[0]);
                (*ibox_ref).add(&image);
                let window_ref = (*ibox_ref).get_parent().unwrap();
                window_ref.show_all();
            });

            hbox.add(&button_reject);
            window.show_all()

        });

        self.app.run(&[]);
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();

    let mut image_files : Vec<PathBuf> = vec!();
    
    if args.len() < 3 {
        println!("Usage: explorer <path to directory of tiff files> <output dir>"); 
        process::exit(1);
    }

    let paths = fs::read_dir(Path::new(&args[1])).unwrap();

    for path in paths {
        match path {
            Ok(file) => {
                let filename = file.file_name();
                let tx = filename.to_str().unwrap();
                if tx.contains("tif") {
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
       
    }

    gtk::init().expect("Unable to start GTK3");
    let app = Explorer::new(image_files, PathBuf::from(&args[2]));
    app.run(app.clone());
}