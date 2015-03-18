---
title: Virtualisation
---

## Virtualisation

### Reproducible research

The ability to reproduce the analyses of research studies is increasingly recognised as important.

Several approaches have developed that help researchers to package up code so that their analyses can be distributed and run by others.

### Virtual machines

A popular method for creating a shareable environment is with the use of virtual machines. 

The isolated system created by virtual machines can be beneficial, but criticisms include:

- size: virtual machines can be bulky
- performance: virtual machines may use significant system resources

Tools such as Vagrant have helped to simplify the process of creating and using virtual machines:  
[https://www.vagrantup.com](https://www.vagrantup.com/)

### Virtual environments

Virtual environments offer an alternative to virtual machines. Rather than constructing an entirely new system, virtual environments in general seek to 
provide reproducible 'containers' which are layered on top of an existing environment.

A popular tool for creating virtual environments is Docker:  
[https://www.docker.com](https://www.docker.com/)

<!-- 
https://github.com/idekerlab/cyREST/wiki/Docker-and-Data-Analysis
http://arxiv.org/pdf/1410.0846v1.pdf
-->


