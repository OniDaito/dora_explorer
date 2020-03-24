/// A small program that lets us find the
/// region of interest in a tiff stack of
/// Dora images
///
/// https://github.com/image-rs/image
///
/// Author: Benjamin Blundell
/// Email: me@benjamin.computer

extern crate rand;
extern crate image;
extern crate nalgebra as na;
extern crate probability;
extern crate scoped_threadpool;
extern crate ndarray;
extern crate tiff;
extern crate fitrs;

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
use image::{ImageBuffer, GrayImage, DynamicImage, imageops};
use std::cmp::Ordering;
use std::f32::consts::*;
use dora_explorer::dora_tiff::get_image;
use dora_explorer::dora_tiff::save_fits;
use dora_explorer::dora_tiff::save_tiff_stack;

/// Points we are interested in
#[derive(Eq)]
struct ImgPoint {
    x : usize,
    y : usize,
}

impl Ord for ImgPoint {
    fn cmp(&self, other: &ImgPoint) -> Ordering {
        if self.y != other.y { 
         return self.y.cmp(&other.y);
        }
        self.x.cmp(&other.x)
    }
}

impl PartialOrd for ImgPoint {
    fn partial_cmp(&self, other: &ImgPoint) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl PartialEq for ImgPoint { 
    fn eq(&self, other: &ImgPoint) -> bool {
        self.y == other.y && self.x == other.x
    }
}

/// Area of interest as a Box 
struct AOIBox {
    x : usize,
    y : usize,
    w : usize,
    h : usize,
}

trait InBox {
    fn inside(&self, x: usize, y: usize) -> bool;
}

impl InBox for AOIBox { 
    fn inside(&self, x : usize, y : usize) -> bool {
        x >= self.x && x<= self.x + self.w && y >= self.y && y <= self.y + self.h  
    }
}

/// Function that looks at the previous scanline, directly above to see if there is a group to merge.
/// We need to use lifetimes to make sure our references to ImgPoint don't disappear on us.
fn lookabove<'a>(groups : &mut Vec< Vec< &'a ImgPoint> >, new_group : &mut Vec<&'a ImgPoint>, cpoint :  &ImgPoint ) {
    if cpoint.y > 0 {
        let mut bdel : bool = false;
        let mut gdel : usize = 0;
        let mut j : usize = 0;

        // We don't really need to name the loops here but it's nice to
        'outer: while j < groups.len() && bdel == false {
            let iter = groups[j].iter().rev();
           
            'inner: for x in iter {
                let prev_point : &'a ImgPoint = x;
                
                if prev_point.x == cpoint.x && prev_point.y == cpoint.y - 1 {
                    bdel = true;
                    gdel = j;
                    break 'outer;
                }

                if prev_point.y < (cpoint.y - 1) || (prev_point.y == (cpoint.y -1) && prev_point.x < cpoint.x){
                    break 'inner;
                }

            }
            j = j + 1;
        }

        if bdel {
            new_group.append(&mut groups[gdel]);
            // Need this sort here to make sure our merged groups are still in order.
            // We really need to PREPEND not append! That would be faster
            new_group.sort(); 
            groups.remove(gdel);
        }
    }
}


fn points_to_boxes (points : Vec<ImgPoint>, width : usize, height : usize, verbose : bool ) -> Vec<AOIBox> {
    let groups = floodfill(&points, verbose);
    let mut boxes : Vec<AOIBox> = vec![];
    
    for group in groups {
        let new_box = area_to_box(group, width, height);
        boxes.push(new_box);
    }
    
    // Limit scope of the 'boxes borrow'
    {
        if verbose {
            for ri in 0..boxes.len() {
                let rb : &AOIBox =  &boxes[ri];
                println!("Group Box: {},{},{},{}", rb.x, rb.y, rb.w, rb.h);
            }
        }
    }
    return boxes;
}


/// Perform a scanline floodfill - https://en.wikipedia.org/wiki/Flood_fill
/// Roughly based on the above but our algorithm looks for connected points
/// by marching along the line checking the pixel above and the pixel to the
/// right, merging groups together. This assumes that points are presented
/// in order of ascending y then ascending x.

fn floodfill ( points : &Vec<ImgPoint>, verbose : bool) -> Vec< Vec< &ImgPoint> > {
    let mut groups : Vec< Vec< &ImgPoint> > = Vec::new();
    
    // Progress bar status
    let mut pb = ProgressBar::new(points.len() as u64);
    pb.format("╢▌▌░╟");

    let mut i: usize = 0;
    if points.len() > 0 {
        while i < points.len() - 1 {
            let mut cpoint : & ImgPoint = &points[i];
            let mut new_group : Vec<& ImgPoint> = vec![&points[i]];

            // Look above the current point - should only be one group
            lookabove(&mut groups, &mut new_group, cpoint);

            // Look to the right of the current point
            // We stop when we have finished with this scanline or it stops being connected
            let cscan : usize = cpoint.y;
            if i < points.len() - 1 {
                while i < points.len() - 1 && points[i+1].x == points[i].x + 1 && cpoint.y == cscan {
                    cpoint = &points[i+1];
                    new_group.push(cpoint);
                    lookabove(&mut groups, &mut new_group, cpoint);
                    i = i + 1;
                    if verbose { pb.inc(); }
                }
            }

            groups.push(new_group);
            i = i + 1; 
            if verbose { pb.inc(); }
        }
    }

    if verbose {
        pb.finish_print("done");
        println!("Groups found: {}", groups.len());
    }
    return groups
}

/// Convert our group of points to a bounding box
fn area_to_box (group : Vec< &ImgPoint>, iw : usize, ih : usize ) -> AOIBox {   
    let mut min_x : usize = iw;
    let mut min_y : usize = ih;
    let mut max_x : usize = 0;
    let mut max_y : usize = 0;

    for point in group {
        if point.x > max_x { max_x = point.x }
        if point.y > max_y { max_y = point.y }
        if point.x < min_x { min_x = point.x }
        if point.y < min_y { min_y = point.y } 
    }

    let new_box = AOIBox {
        x : min_x,
        y : min_y,
        w : max_x - min_x,
        h : max_y - min_y,
    };

    return new_box;
}

/// Gaussian function for our pixel selection. 
/// TODO - we don't really need to use the full equation. Could cache some 
/// common values here for not a lot of memory and speed things up.
fn gg (x : f32, v : f32, u : f32) -> f32 {
    let a : f32 = 1.0 / (v * (2.0 * PI).sqrt());
    let b : f32 = (((x - u) / v).powi(2)) * -0.5;
    E.powf(b) * a
}

fn kn (x : usize, y : usize, k : usize, img : &Vec<Vec<f32>> ) -> f32 { 
    let height = img.len() as i32;
    let width = img[0].len() as i32;
    let mut val : f32 = 0.0;
    let uk = k as i32;
    let ux = x as i32;
    let uy = y as i32;
    
    for i in 0..uk {
        for j in 0..uk {
            let xoff = (j as i32) - uk / 2;
            let yoff = (i as i32) - uk / 2;

            let fx = ux + xoff;
            let fy = uy + yoff;

            if fx >= 0 && fy >= 0 && fx < width && fy < height {
                let nx = fx as usize;
                let ny = fy as usize;

                let pixel = img[ny][nx] as f32; // Was previously normalised
                let dd : f32 = ((xoff as f32).abs() * (yoff as f32).abs()).sqrt(); 

                val = val + (gg(dd, 2.0, 0.0) * pixel);
            }
        }
    }
    val
}

/// Given an image, create a set of groups of points that pass our test.
/// We consider a 3x3 patch, take gaussians of each and sum to see if it's
/// over the threshold
fn create_points(img : &Vec<Vec<f32>>, verbose : bool) -> Vec<ImgPoint> {
    // First find a good threshold - mean score
    let height = img.len();
    let width = img[0].len();
    let mut avg : f32 = 0.0;
    let image_size = (width * height) as f32;
    
    for y in 0..height {
        for x in 0..width{
            let pixel = img[y][x];
            avg = avg + (pixel as f32);
        }
    }

    avg = avg / image_size;
    if verbose { println!("Average value: {}", avg); }

    // Variance and Standard Deviation
    let mut var : f32 = 0.0;
    for y in 0..height {
        for x in 0..width{
            let pixel = img[y][x] as f32;
            let dif = (avg - pixel) * (avg - pixel);
            var = var + dif;
        }
    }
    var = var / image_size;
    
    let sdd : f32 = var.sqrt();
    let level : f32 = avg + (sdd * 2.0);
    let kernel_size : usize = 5; // 5 seems pretty good on most examples
    if verbose { println!("Variance / Std. Dev. : {}, {}", var, sdd); }

    // Now find all the points of interest over 2 standard deviations away.
    let mut points : Vec<ImgPoint> = Vec::new();
    for y in 0..height {
        for x in 0..width{
            //let pixel = img.get_pixel(x, y)[0] as f32;
            let mut final_val = kn(x, y, kernel_size, &img); // Assume 0 to 1.0 normed
            
            if final_val > level {
                let mut new_point = ImgPoint {
                    x : x,
                    y : y,
                };
                points.push(new_point); 
            }
        }
    }

    points.sort();
    
    if verbose {
        let pip : f32 = (points.len() as f32) / image_size * 100.0;
        println!("Points of interest %: {}", pip);
    }
    
    return points;
}

pub fn equal_rows(input_img : &Vec<Vec<u16>>) -> bool {
    let mut ll : usize = input_img[0].len();
    for row in input_img {
        if row.len() != ll {
            return false;
        }
        ll = row.len();
    }
    return true;
}

// Given an input image and a new, smaller image, replace the corresponding pixels, using
// top and left.
pub fn replace(input_img: &Vec<Vec<u16>>, replace_img: &Vec<Vec<u16>>, top : usize, left : usize) -> Vec<Vec<u16>> {
    let mut new_image : Vec<Vec<u16>> = vec![];
    let height = input_img.len();
    let width = input_img[0].len();
    let nh = replace_img.len();
    let nw = replace_img[0].len();

    for y in 0..height{
        let mut new_row : Vec<u16> = vec![];
        
        for x in 0..width {
            if x >= left && x < left + nw &&
                y >= top && y < top + nh && nw > 0 && nh > 0 {    
                new_row.push(replace_img[y-top][x-left]);
            } else {
                new_row.push(input_img[y][x]);
            }
        }
        new_image.push(new_row);
    }

    return new_image;
}


// Given top, left, width and height, crop our raw image down, returning a new sub image.
pub fn crop(raw_img: &Vec<Vec<u16>>, top : usize, left: usize, width : usize, height : usize) -> Vec<Vec<u16>> {
    let mut sub_image : Vec<Vec<u16>> = vec![];
    assert_eq!(equal_rows(&raw_img), true);

    for y in 0..raw_img.len() {
        if y >= top && y < top + height {
            let mut new_row : Vec<u16> = vec![];
            for x in 0..raw_img[y].len() {
                if x >= left && x < left + width {
                    new_row.push(raw_img[y][x]);
                }
            }
            sub_image.push(new_row);
        }
    }
    return sub_image;
}

// Given our float image, look for ROIs. If we have some, choose the biggest and use that
// to crop down the raw_img.
pub fn roi(float_img : &Vec<Vec<f32>>, raw_img: &Vec<Vec<u16>>, width : usize, height : usize) -> Vec<Vec<u16>> {
    let verbose = false;
    let boxes = points_to_boxes(create_points(&float_img, verbose), width, height, verbose);
    let size : usize = width * height;

    assert_eq!(equal_rows(&raw_img), true);

    let mut max_w : usize = usize::min_value();
    let mut min_w : usize = usize::max_value();
    let mut max_h : usize = usize::min_value();
    let mut min_h : usize = usize::max_value();
 
    // Null image, if we get no ROIs
    let mut null_img : Vec<Vec<u16>> = vec![];
    let mut null_img_row : Vec<u16> = vec![];
    null_img_row.push(0u16);
    null_img.push(null_img_row);

    // If none found then return the null image
    if boxes.len() == 0 {
        return null_img;
    }
    // We look at the min max extents to build the largest box
    // Sort of defeats the point a little but for now it's ok
   
    for i in 0..boxes.len() {
        if boxes[i].x < min_w { min_w = boxes[i].x; }
        if boxes[i].x + boxes[i].w > max_w { max_w = boxes[i].x + boxes[i].w; }
        
        if boxes[i].y < min_h { min_h = boxes[i].y; }
        if boxes[i].y + boxes[i].h  > max_h { max_h = boxes[i].y + boxes[i].h; }
    }

    // Now square off the box
    //println!("{},{}", min_h, max_h);
    let mut mh : usize = max_h - min_h; 
    let mut mw : usize = max_w - min_w;
   
    if mh >= 16 && mw >= 16 {    
        let figimg = crop(&raw_img, min_h, min_w, mw, mh);
        //let figimg = crop(&raw_img, 0, 0, width, height);
        assert_eq!(equal_rows(&figimg), true);
        return figimg;
    } else {
      return null_img;
    }
}

// Given a decoded tiff, return a normalised f32 vector vector for detecing our regions of 
// interest and the original image as an image::DynamicImage struct for further processing.
pub fn normed(input_image : Vec<u16>, width : usize, height : usize) -> (Vec<Vec<f32>>, Vec<Vec<u16>>) {
    let mut float_image : Vec<Vec<f32>> = vec![];
        
    for y in 0..height {
        let mut row  : Vec<f32> = vec![];
        for x in 0..width {
            row.push(0 as f32);
        }
        float_image.push(row);
    }

    // Normalise for better work with the gaussian bit. 
    // Find the min max extents
    let mut min_val : f32 = 1e10;
    let mut max_val : f32 = 0.0;

    for y in 0..height {
        for x in 0..width {
            let vv = input_image[y * height + x] as f32;
            if vv > max_val { max_val = vv };
            if vv < min_val { min_val = vv };
        }
    } 
  
    for y in 0..height {
        for x in 0..width {
            float_image[y][x] = ((input_image[y * height + x] as f32) - min_val) / (max_val - min_val);
        }
    }

    // Keep the raw image but re-index for ease of use later
    let mut raw_image : Vec<Vec<u16>> = vec![];
    for y in 0..height {
        let new_row : Vec<u16> = vec![];
        raw_image.push(new_row);
        for x in 0..width {
            raw_image[y].push(input_image[y * height + x]);
        }
    }

    return (float_image, raw_image)
}

pub fn tiff_to_vec(path : &Path) -> (Vec<Vec<Vec<u16>>>, usize, usize) {
    let img_file = File::open(path).expect("Cannot find test image!");
    let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

    assert_eq!(decoder.colortype().unwrap(), ColorType::Gray(16));
    let img_res = decoder.read_image().unwrap();

    // Check the image size here
    let (w, h) = decoder.dimensions().unwrap();
    let width = w as usize;
    let height = h as usize;

    // Our buffer - we sum all the image here and then scale
    let mut img_buffer : Vec<Vec<Vec<u16>>> = vec![];    
    let mut levels : usize = 0;

    // Now we've decoded, lets update the img_buffer
    if let DecodingResult::U16(img_res) = img_res {
        let (first_layer, first_raw_image) = normed(img_res, width, height);
        img_buffer.push(roi(&first_layer, &first_raw_image, width, height));

        while decoder.more_images() {
           
            let next_res = decoder.next_image();
            match next_res {
                Ok(res) => {   
                    let img_next = decoder.read_image().unwrap();
                    if let DecodingResult::U16(img_next) = img_next {
                        levels += 1;
                        
                        let (next_layer, next_raw_image) = normed(img_next, width, height);
                        img_buffer.push(roi(&next_layer, &next_raw_image, width, height));
                    }
                },
                Err(_) => {}
            }
        }
    }

    // Now find the max size in the image. We'll resize by adding a 
    // black border of equal size around it.
    // essentially all sizes of ROI are centered in the middle of the
    // image as best as we can.
    let mut maxw : usize = 0;
    let mut maxh : usize = 0;
    let mut maxs : usize = 0;
    for img in &img_buffer {
        let w = img[0].len();
        let h = img.len();
        if w as usize > maxw { maxw = w as usize; }
        if h as usize > maxh { maxh = h as usize; }
    }

    maxs = maxw; if maxh > maxs { maxs = maxh; }
    //assert_eq!(maxw == maxh, true); // Must be square
    let mut fimg_buffer : Vec<Vec<Vec<u16>>> = vec![];

    for img in &img_buffer {
       
        let iw = img[0].len();
        let ih = img.len();
        let nx : usize = (maxs - iw) / 2;
        let ny : usize = (maxs - ih) / 2;

        let mut luma_ex : Vec<Vec<u16>> = vec![];
        for y in 0..maxs {
            let new_row : Vec<u16> = vec![];
            luma_ex.push(new_row);
            for x in 0..maxs {
                luma_ex[y].push(0u16);
            }
        } 
        let final_ex = replace(&luma_ex, &img, nx, ny);
        fimg_buffer.push(final_ex);
    }

    (fimg_buffer, maxs, maxs)
}

fn process_tiffs (image_paths : &Vec<PathBuf>, out_path : &String,  nthreads : u32) {
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
                    let (timg, width, height) = tiff_to_vec(&cslice[_i]);
                    let fidx = format!("/image_{:06}.tiff", ((start + _i) * 4) as usize);
                    let mut tiffpath = out_path.clone();
                    tiffpath.push_str(&fidx);
                    save_tiff_stack(&timg, &tiffpath, width, height);

                    // TODO - fits or tiff?
                    //save_fits(&timg, &fitspath, rwidth, rheight);

                    // Now Augment - eventually
                    /*
                    let fidx1 = format!("/image_{:06}.fits", ((start + _i) * 4 + 1) as usize);
                    fitspath = out_path.clone();
                    fitspath.push_str(&fidx1);
                    let aimg1 = aug_vec(&timg, Direction::Right);
                    save_fits(&aimg1, &fitspath, width, height);

                    let fidx2 = format!("/image_{:06}.fits", ((start + _i) * 4 + 2) as usize);
                    fitspath = out_path.clone();
                    fitspath.push_str(&fidx2);
                    let aimg2 = aug_vec(&timg, Direction::Down);
                    save_fits(&aimg2, &fitspath, width, height);

                    let fidx3 = format!("/image_{:06}.fits", ((start + _i) * 4 + 3) as usize);
                    fitspath = out_path.clone();
                    fitspath.push_str(&fidx3);
                    let aimg3 = aug_vec(&timg, Direction::Right);
                    save_fits(&aimg3, &fitspath, width, height);*/
            
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
            <output dir> <num threads> <OPTIONAL filter>"); 
        process::exit(1);
    }

    let mut filter : String = String::new();
    if args.len() == 5 {
        filter.push_str(&args[4]);
    }

    let paths = fs::read_dir(Path::new(&args[1])).unwrap();
    let nthreads = args[3].parse::<u32>().unwrap();
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
    image_files.sort();
    process_tiffs(&image_files, &args[2], nthreads);
}