# dora_explorer

A couple of programs for dealing with Dora's data. This is written in rust and requires a few external libraries. This project creates two programs. One lets us view a directory of 16bit tiff images with some augmentation. The other processes each stack to produce the data we need.

## Requirements

* Rust crates such as tiff and fitrs, though these are installed easily with the build and run commands.
* gtk+ development libraries so that the rust crate gtk-rs can build.

## Building

If you have rust installed, enter the swiss_parse directory and type

    cargo build

I think it's a good idea to make sure your rust compilier is up-to-date. I've fallen afoul of this a few times

    rustup upgrade

## Process

The program *explorer* views the tiff files and spits out fits format images. The program process does the same but a tad faster with no GTK window. The programs are invoked as follows:

    cargo run --release --bin explorer <path to dora tiffs> <path to save fits>
    cargo run --release --bin process <path to dora tiffs> <path to save fits> <num threads>

### Image processing

Dora's tiffs appear to be 3D; stacks of about 41 images long, in 16 bit greyscale. We change these to a mean average, adding up every pixel and dividing by the number of layers in the Z. We save as floating point as that's a bit easier to mess with further down the line.