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

    use gtk::prelude::*;
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
    use tiff::ColorType;
    use fitrs::{Fits, Hdu};

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

    pub fn aug_vec(img : &Vec<Vec<f32>>, dir : Direction) -> Vec<Vec<f32>> {
        let mut new_img : Vec<Vec<f32>> = vec!();
        let mut rm : [[i32; 2]; 2] =  [[0, -1],[1, 0]];
        match dir { // *self has type Direction
            Direction::Right => {
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
                let nx = (x as i32 * rm[0][0] + y as i32 * rm[0][1]) as usize;
                let ny = (x as i32 * rm[1][0] + y as i32 * rm[1][1]) as usize;
                new_img[y][x] = img[y][x];
             }
        }

        new_img
    }

    pub fn tiff_to_vec(path : &Path, width: usize, height: usize) -> (Vec<Vec<f32>>, f32, f32, usize) {
        let img_file = File::open(path).expect("Cannot find test image!");
        let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

        let mut minp : f32 = 1e12; // we might end up overflowing!
        let mut maxp : f32 = 0.0;

        assert_eq!(decoder.colortype().unwrap(), ColorType::Gray(16));
        let img_res = decoder.read_image().unwrap();

        // Check the image size here

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
            for y in 0..height as usize {
                for x in 0..width as usize {
                    img_buffer[y][x] = (img_res[y * (height as usize) + x] as f32);
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

        (img_buffer, minp, maxp, levels)
    }

    // Convert our model into a gtk::Image that we can present to
    // the screen.
    pub fn get_image(path : &Path,  width: usize, height: usize) -> (gtk::Image, Vec<Vec<f32>>) {
        let (img_buffer, minp, maxp, levels) = tiff_to_vec(path, width, height);
        let mut final_buffer : Vec<u8> = vec![];

        for y in 0..height {
            for x in 0..width  {
                let colour = (img_buffer[y][x] / maxp * 255.0) as u8;
                let idx = (y * (height) + x) * 3;
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
        (image, img_buffer)
    }

    // Save out the FITS image
    pub fn save_fits(img : &Vec<Vec<f32>>, filename : &String, width: usize, height : usize) {
        let mut data : Vec<f32> = (0..height)
            .map(|i| (0..width).map(
                move |j| (i + j) as f32)).flatten().collect();

        for _y in 0..height {
            for _x in 0..width {
                let idx : usize = (_y * width +_x ) as usize; 
                data[idx] = img[_x as usize][(height - _y - 1) as usize];
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
}

