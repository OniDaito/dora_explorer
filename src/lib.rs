/// Library for Dora's stuff.
///
/// Author: Benjamin Blundell
/// Email: me@benjamin.computer

extern crate rand;
extern crate image;
extern crate ndarray;
extern crate gtk;
extern crate gio;
extern crate gdk_pixbuf;
extern crate glib;
extern crate tiff;
extern crate fitrs;

pub mod dora_tiff {
    use gio::prelude::*;
    use gdk_pixbuf::Pixbuf;
    use gdk_pixbuf::Colorspace;
    use glib::Bytes;
    use std::fmt;
    use std::fs;
    use std::fs::{File, OpenOptions};
    use std::io::prelude::*;
    use std::path::{Path, PathBuf};
    use tiff::decoder::{ifd, Decoder, DecodingResult};
    use image::{GrayImage, DynamicImage, imageops};
    use tiff::ColorType;
    use tiff::encoder::*;
    use fitrs::{Fits, Hdu};
    use std::f32::consts::PI;

    // For augmentation, we have 90, 180 and 270 degrees to avoid artefacts
    pub enum Direction {
       Right,
       Down,
       Left
    }

    pub fn check_size(path : &Path, width: usize, height: usize) -> bool {
        let img_file = File::open(path).expect("Cannot find test image!");
        let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
        let (dims_width, dims_height) = decoder.dimensions().unwrap();
        dims_height as usize == height && dims_width as usize == width
    }

    pub fn get_size(path : &Path) -> (usize, usize, usize) {
        let img_file = File::open(path).expect("Cannot find test image!");
        let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");
        let (dims_width, dims_height) = decoder.dimensions().unwrap();
        let mut dims_depth : usize = 0;
        
        while decoder.more_images() {
            dims_depth += 1;
            let next_res = decoder.next_image();
        }
        
        (dims_height as usize,  dims_width as usize, dims_depth as usize)
    }

    pub fn aug_vec(img : &Vec<Vec<f32>>, dir : Direction) -> Vec<Vec<f32>> {
        let mut new_img : Vec<Vec<f32>> = vec!();
        let height = img.len() as i32;
        let width = img[0].len() as i32;
        let mut rm : [[i32; 2]; 2] = [[0, -1],[1, 0]];

        for y in 0..img.len() {
            let mut row : Vec<f32> = vec!();
            for x in 0..img[0].len() {
                row.push(0.0);
            }
            new_img.push(row);
        }
        match dir { // *self has type Direction
            Direction::Right => {
                rm = [[0, -1],[1, 0]];
            },
            Direction::Down =>{
                rm = [[-1, 0],[0, -1]];
            },
            Direction::Left => {
                rm = [[0, 1],[-1, 0]];
            },
        }

        for y in 0..img.len() {
            for x in 0..img[0].len() {

                match dir { // *self has type Direction
                    Direction::Right => {
                        let mut nx = (width - 1) + (x as i32 * rm[0][0] + y as i32 * rm[0][1]);
                        let mut ny = (x as i32 * rm[1][0] + y as i32 * rm[1][1]);
                        new_img[y][x] = img[ny as usize][nx as usize];
                    },
                    Direction::Down =>{
                        let mut nx = (width - 1) + (x as i32 * rm[0][0] + y as i32 * rm[0][1]);
                        let mut ny = (height - 1) + (x as i32 * rm[1][0] + y as i32 * rm[1][1]);
                        new_img[y][x] = img[ny as usize][nx as usize];
                    },
                    Direction::Left => {
                        let mut nx = (x as i32 * rm[0][0] + y as i32 * rm[0][1]);
                        let mut ny = (height - 1) + (x as i32 * rm[1][0] + y as i32 * rm[1][1]);
                        new_img[y][x] = img[ny as usize][nx as usize];
                    },
                }
            }
        }

        new_img
    }

    pub fn aug_stack(stack : &Vec<Vec<Vec<u16>>>, dir : Direction) -> Vec<Vec<Vec<u16>>> {
        let mut new_stack : Vec<Vec<Vec<u16>>> = vec!();
        let mut rm : [[i32; 2]; 2] = [[0, -1],[1, 0]];
        let height = stack[0].len() as i32;
        let width = stack[1].len() as i32;
        
        for z in 0..stack.len(){
            let mut new_img : Vec<Vec<u16>> = vec!();
            for y in 0..stack[0].len() {
                let mut row : Vec<u16> = vec!();
                for x in 0..stack[0][0].len() {
                    row.push(0u16);
                }
                new_img.push(row);
            }
            new_stack.push(new_img);
        }

        match dir { // *self has type Direction
            Direction::Right => {
                rm = [[0, -1],[1, 0]];
            },
            Direction::Down =>{
                rm = [[-1, 0],[0, -1]];
            },
            Direction::Left => {
                rm = [[0, 1],[-1, 0]];
            },
        }

        for z in 0..stack.len() {
            let img = &stack[z];
            for y in 0..img.len() {
                for x in 0..img[0].len() {

                    match dir { // *self has type Direction
                        Direction::Right => {
                            let mut nx = (width - 1) + (x as i32 * rm[0][0] + y as i32 * rm[0][1]);
                            let mut ny = (x as i32 * rm[1][0] + y as i32 * rm[1][1]);
                            new_stack[z][y][x] = img[ny as usize][nx as usize];                       
                        },
                        Direction::Down =>{
                            let mut nx = (width - 1) + (x as i32 * rm[0][0] + y as i32 * rm[0][1]);
                            let mut ny = (height - 1) + (x as i32 * rm[1][0] + y as i32 * rm[1][1]);
                            new_stack[z][y][x] = img[ny as usize][nx as usize];                       
                        },
                        Direction::Left => {
                            let mut nx = (x as i32 * rm[0][0] + y as i32 * rm[0][1]);
                            let mut ny = (height - 1) + (x as i32 * rm[1][0] + y as i32 * rm[1][1]);
                            new_stack[z][y][x] = img[ny as usize][nx as usize];                       
                        },
                    }

                }
            }
        }

        new_stack
    }

    pub fn tiff_to_vec(path : &Path) -> (Vec<Vec<f32>>, usize, usize, f32, f32, usize) {
        let img_file = File::open(path).expect("Cannot find test image!");
        let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

        let mut minp : f32 = 1e12; // we might end up overflowing!
        let mut maxp : f32 = 0.0;

        assert_eq!(decoder.colortype().unwrap(), ColorType::Gray(16));
        let img_res = decoder.read_image().unwrap();

        // Check the image size here
        let (w, h) = decoder.dimensions().unwrap();
        let width = w as usize;
        let height = h as usize;

        // Our buffer - we sum all the image here and then scale
        let mut img_buffer : Vec<Vec<f32>> = vec![];
        for y in 0..height {
            let mut row  : Vec<f32> = vec![];
            for x in 0..width {
                row.push(0 as f32);
            }
            img_buffer.push(row);
        }

        // Final buffer that we use that is a little smaller - u8
        // and not u16, but also RGB, just to make GTK happy.
        let mut final_buffer : Vec<u8> = vec![];
        for y in 0..height {
            let mut row  : Vec<u8> = vec![];
            for x in 0..width {
                // GTK insists we have RGB so we triple everything :/
                for _ in 0..3 {
                    final_buffer.push(0 as u8);
                }
            }
        }
        
        let mut levels : usize = 0;

        // Now we've decoded, lets update the img_buffer
        if let DecodingResult::U16(img_res) = img_res {
            for y in 0..height {
                for x in 0..width {
                    img_buffer[y][x] = (img_res[y * height + x] as f32);
                }
            }

            while decoder.more_images() {
                let next_res = decoder.next_image();
                match next_res {
                    Ok(res) => {   
                        let img_next = decoder.read_image().unwrap();
                        if let DecodingResult::U16(img_next) = img_next {
                            levels += 1;
                            for y in 0..height as usize {
                                for x in 0..width as usize {
                                    img_buffer[y][x] += (img_next[y * (height as usize) + x] as f32);
                                }
                            } 
                        }
                    },
                    Err(_) => {}
                }
            }
            // We take an average rather than a total sum
            for y in 0..height as usize {
                for x in 0..width as usize {
                    img_buffer[y][x] = img_buffer[y][x] / (levels as f32);
                }
            }

            // Find min/max
           
            for y in 0..height as usize {
                for x in 0..width as usize {
                    if (img_buffer[y][x] as f32) > maxp { maxp = img_buffer[y][x] as f32; }
                    if (img_buffer[y][x] as f32) < minp { minp = img_buffer[y][x] as f32; }
                }
            }
        }

        (img_buffer, width, height, minp, maxp, levels)
    }

    // Perform a gauss blur
    pub fn gauss_blur(img : &Vec<Vec<f32>>, gauss : f32 ) -> Vec<Vec<f32>> {
        // http://blog.ivank.net/fastest-gaussian-blur.html
        let rs = (gauss * 2.57).ceil() as usize;
        let height = img.len();
        let width = img[0].len();
        
        // New temp image
        let mut img_blurred : Vec<Vec<f32>> = vec![];

        for y in 0..height {
            let mut row : Vec<f32> = vec!();
            for x in 0..width {
                row.push(0f32);
            }
            img_blurred.push(row);
        }

        for h in 0..height {
            for w in 0..width {
                let mut val : f32 = 0.0;
                let mut wsum : f32 = 0.0;

                for i in 0..(rs*2+1) {
                    let iy : f32 = (h as f32 ) - (rs as f32) + (i as f32);

                    for j in 0..(rs*2+1) {
                        let ix : f32 = (w as f32 ) - (rs as f32) + (j as f32);

                        let x = ((width - 1) as f32).min(0f32.max(ix)) as usize;
                        let y = ((height -1) as f32).min(0f32.max(iy)) as usize;
                        let dsq = (ix - w as f32) * (ix - w as f32) + (iy - h as f32) * (iy - h as f32);
                        let wght = (-dsq / (2.0*gauss*gauss)).exp() / (PI * 2.0 * gauss * gauss);
                        val += img[y][x] * wght;
                        wsum += wght;
                    }
                }
                img_blurred[h][w] = val / wsum;
            }
        }
        img_blurred
    }


    pub fn tiff_to_stack(path : &Path) -> (Vec<Vec<Vec<u16>>>, usize, usize, usize) {
        let img_file = File::open(path).expect("Cannot find test image!");
        let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

        assert_eq!(decoder.colortype().unwrap(), ColorType::Gray(16));
        let img_res = decoder.read_image().unwrap();

        // Check the image size here
        let (w, h) = decoder.dimensions().unwrap();
        let width = w as usize;
        let height = h as usize;

        let mut img_stack : Vec<Vec<Vec<u16>>> = vec![];

        // Our buffer - we sum all the image here and then scale
        let mut img_buffer : Vec<Vec<u16>> = vec![];

        // Now we've decoded, lets update the img_buffer
        if let DecodingResult::U16(img_res) = img_res {
            for y in 0..height {
                let mut row : Vec<u16> = vec![];
                for x in 0..width {
                    row.push((img_res[y * height + x] as u16));
                }
                img_buffer.push(row);
            }

            img_stack.push(img_buffer);

            while decoder.more_images() {
                let next_res = decoder.next_image();
                match next_res {
                    Ok(res) => {   
                        let img_next = decoder.read_image().unwrap();
                        if let DecodingResult::U16(img_next) = img_next {
                            let mut next_buffer : Vec<Vec<u16>> = vec![];

                            for y in 0..height as usize {
                                let mut row : Vec<u16> = vec![];
                                for x in 0..width as usize {
                                    row.push(img_next[y * (height as usize) + x] as u16);
                                }
                                next_buffer.push(row);
                            }
                            img_stack.push(next_buffer);
                        }
                    },
                    Err(_) => {}
                }
            }
        }
        let depth = img_stack.len();
        (img_stack, width, height, depth)
    }

    // Convert our model into a gtk::Image that we can present to
    // the screen.
    pub fn get_image(path : &Path) -> (gtk::Image, Vec<Vec<f32>>, usize, usize) {
        let (img_buffer, width, height, minp, maxp, levels) = tiff_to_vec(path);
        let mut final_buffer : Vec<u8> = vec![];

        for x in 0..width {
            for y in 0..height {
                final_buffer.push(0 as u8);
                final_buffer.push(0 as u8);
                final_buffer.push(0 as u8);
            }
        }

        for x in 0..width  {
            for y in 0..height {
                let colour = (img_buffer[x][y] / maxp * 255.0) as u8;
                let idx = (x * (width) + y) * 3;
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
            width as i32,
            height as i32,
            (width * 3 * 1) as i32
        );

        let image : gtk::Image = gtk::Image::new_from_pixbuf(Some(&pixybuf));
        (image, img_buffer, width, height)
    }

    // Save out the FITS image
    pub fn save_fits(img : &Vec<Vec<f32>>, filename : &String, width: usize, height : usize) {
        let mut data : Vec<f32> = (0..height)
            .map(|i| (0..width).map(
                move |j| (i + j) as f32)).flatten().collect();

        for _x in 0..width {
            for _y in 0..height {
                let idx : usize = (_x * width +_y ) as usize; 
                data[idx] = img[_x as usize][_y as usize];
                // / intensity * MULTFAC;
            }
        }

        let mut primary_hdu = 
            Hdu::new(&[width as usize , height as usize], data);
        // Insert values in header
        primary_hdu.insert("NORMALISATION", "NONE");
        primary_hdu.insert("WIDTH", width as i32);
        primary_hdu.insert("HEIGHT", height as i32);
        Fits::create(filename, primary_hdu).expect("Failed to create");  
    }

    // Save out the TIFF stack as 16bit greyscale
    pub fn save_tiff_stack(img_stack : &Vec<Vec<Vec<u16>>>, filename : &String, width: usize, height : usize) {
        let mut file = File::create(filename).unwrap();
        let mut tiff = TiffEncoder::new(&mut file).unwrap();
        
        for img in img_stack {
            let mut b : Vec<u16> = vec![];
            for y in 0..img.len(){
                for x in 0..img[y].len(){
                    b.push(img[y][x]);
                }
            }
            let slice = &b[..];
            tiff.write_image::<colortype::Gray16>(width as u32, height as u32, &slice).unwrap();
        }
    }

    // Save out the stack as float FITS greyscale
    pub fn save_fits_stack(img_stack : &Vec<Vec<Vec<u16>>>, filename : &String, width: usize, height : usize, depth : usize) {
        let mut data : Vec<f32> = Vec::with_capacity(depth * height * width);
        for i in 0..(depth * height * width) {
            data.push(0.0);
        }

        for _x in 0..width {
            for _y in 0..height {
                for _z in 0..depth {
                    let idx : usize = ((_z * width * height) +  (_x * width) +_y ) as usize; 
                    data[idx] = img_stack[_z as usize][_y as usize][_x as usize] as f32;
                }
            }
        }

        let mut primary_hdu =  Hdu::new(&[width as usize , height as usize,  depth as usize], data);
        // Insert values in header
        primary_hdu.insert("NORMALISATION", "NONE");
        primary_hdu.insert("WIDTH", width as i32);
        primary_hdu.insert("HEIGHT", height as i32);
        primary_hdu.insert("DEPTH", depth as i32);
        Fits::create(filename, primary_hdu).expect("Failed to create");  
    }
}

