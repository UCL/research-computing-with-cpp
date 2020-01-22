---
title: VirtualBox 
---

## Running an Ubuntu Desktop as a VM using VirtualBox

### Installing VirtualBox

We will be running a linux virtual machine using VirtualBox software on either your laptop, or if you do not have 
one on one of the Desktop@UCL machines in Physics Lab 1. If you have a laptop this is the preferred option as it 
will mean you will be able to work on the homework problems and coursework without needing access to a cluster room. If you don’t have a laptop then that is fine as well as the whole course can be completed using the Desktop@UCL machines.

Download and install on your laptop:
* The latest VirtualBox from [virtualbox.org](https://www.virtualbox.org/wiki/Downloads) 
* The VirtualBox Extension Pack (from same page) 

If you are using one of the Desktop@UCL machines then VirtualBox is already installed.  

### Creating the VM and installing Ubuntu

Download the [Ubuntu 18.04.3 LTS](https://ubuntu.com/download/desktop) and save the `.iso` file somewhere locally on your laptop or if you are using one of the Desktop@UCL machines someone in your home folder.

Then open VirtualBox and create a new VM of Type: Linux and Version: Ubuntu 64 bit. 2 GB ram is sufficient. 

Choose `Create a virtual hard disk now` and select it to be of type VirtualBox Disk Image (VDI) that is Dynamically allocated. Between 10 GB and 20 GB storage is sufficient. 

If you are using a Desktop@UCL machine then it is important that you save the virtual machine `.vdi` file to your network drive in a folder in `Home (N:)` so that you can access the virtual machine from different Desktop@UCL machines. 

If you are using a different Desktop@UCL machine then each time you open VirtualBox you may need to specify the location
of you VM .vdi file using the `Machine` then `Add` option as opposed to a new machine.   

Then you can start your VM for the first time it will ask you to specify the Ubuntu .iso that you downloaded earlier. Follow the on screen installation instructions. 

When installing Ubuntu select the `Minimal installation` and check the box which says `Download updates while installing Ubuntu`. Select `Erase disk and install Ubuntu` and `Encrypt the new Ubuntu installation` boxes but make sure you write down you security key password as if you lose this you will lose access to the VM.

Once complete you'll need to shutdown and restart the VM.

### Installing required packages

Then you need to install the following packages using the apt-get:
```
sudo apt-get update 

sudo apt-get install build-essential cmake git

sudo apt-get install cmake 

sudo apt-get install git 
```

### Misc

To enable copy and paste between host machine (your laptop) and the VM you may need to complete the following steps:
Power off the VM, then Settings -> Advanced -> and set Shared Clipboard to Bidirectional
Restart the VM and see if you can copy and paste (in Ubuntu paste is `Shift-Ctrl-C` and `Shift-Ctrl-V`).
If you are unable to copy to from your host the the VM then the following may need to be done:

```
sudo apt-get install virtualbox-guest-x11-hwe

sudo VBoxClient --clipboard
```
