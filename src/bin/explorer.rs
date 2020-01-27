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
use std::{cell::Cell, rc::Rc};
use tiff::decoder::{ifd, Decoder, DecodingResult};
use tiff::ColorType;
use std::process;
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
    image_index : Cell<usize> // use this so we can mutate it later
}

// Convert our model into a gtk::Image that we can present to
// the screen.

fn get_image(path : &Path) -> gtk::Image {

    let img_file = File::open(path).expect("Cannot find test image!");
    let mut decoder = Decoder::new(img_file).expect("Cannot create decoder");

    assert_eq!(decoder.colortype().unwrap(), ColorType::Gray(16));
    let img_res = decoder.read_image().unwrap();

    if let DecodingResult::U16(img_res) = img_res {
        let mut levels : usize = 0;
        while decoder.more_images() {
            let img_next_res = decoder.next_image();
            match (img_next_res) {
                Ok(res) => { levels += 1; },
                Err(_) => {}
            }
        }

        println!("Succesfully read {} which has {} levels.", path.display(), levels);

        // Convert down the tiff so we can see it.
        
        /*let pixybuf = Pixbuf::new_from_bytes(&img_res,
            Colorspace::Rgb,
            false, 
            8,
            WIDTH as i32,
            HEIGHT as i32,
            (WIDTH * 3 * 1) as i32
        );

        let image : gtk::Image = gtk::Image::new_from_pixbuf(Some(&pixybuf));
        return image;*/

    } else {
        panic!("Wrong data type");
    }

    let image: gtk::Image = gtk::Image::new();
    image
}


// Our chooser struct/class implementation. Mostly just runs the GTK
// and keeps a hold on our models.
impl Explorer {
    pub fn new(image_paths : Vec<PathBuf>) -> Rc<Self> {
        let app = Application::new(
            Some("com.github.gtk-rs.examples.basic"),
            Default::default(),
        ).expect("failed to initialize GTK application");

        let mut image_index : Cell<usize> = Cell::new(0);

        let explorer = Rc::new(Self {
            app,
            image_paths,
            image_index
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
            let image = get_image(&(app.image_paths[0]));
            //ibox.add(&image);
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

            button_accept.connect_clicked( move |button| {
                println!("Accepted {}", app_accept.image_index.get());
            
                /*let ibox_ref = ibox_accept.lock().unwrap();
                let children : Vec<gtk::Widget> = (*ibox_ref).get_children();
                app_accept.model_index.set(mi + 1);
                let image = get_image(&(app_accept.models[mi + 1]), scale);
                (*ibox_ref).remove(&children[0]);
                (*ibox_ref).add(&image);
                let window_ref = (*ibox_ref).get_parent().unwrap();
                window_ref.show_all();*/
            });

            hbox.add(&button_accept);

            let button_reject = Button::new_with_label("Reject");
            let ibox_reject = ibox_arc.clone();
            let mut app_reject = app.clone();

            button_reject.connect_clicked( move |button| {
                println!("Rejected {}", app_reject.image_index.get());

                // Check we aren't overrunning
                /*let mi = app_reject.model_index.get();
                if mi >= app_reject.models.len() {
                    println!("All models checked!");
                    process::exit(1);
                }

                let ibox_ref = ibox_reject.lock().unwrap();
                let children : Vec<gtk::Widget> = (*ibox_ref).get_children();
                app_reject.model_index.set(mi + 1);
                let image = get_image(&(app_reject.models[mi + 1]), scale);
                (*ibox_ref).remove(&children[0]);
                (*ibox_ref).add(&image);
                let window_ref = (*ibox_ref).get_parent().unwrap();
                window_ref.show_all();*/
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
    
    if args.len() < 2 {
        println!("Usage: explorer <path to directory of tiff files>"); 
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
    let app = Explorer::new(image_files);
    app.run(app.clone());
}