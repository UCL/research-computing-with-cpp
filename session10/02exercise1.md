---
title: Exercise 1 - Working in the cloud
---

## Exercise 1: Working in the cloud

### Spinning up an instance

This example briefly run through the process of:

- creating an account with a provider of cloud services
- spinning up a single cloud instance using a web interface

### Create an account

A growing number of companies are offering cloud computing services, for example: 

- Amazon Web Services: [http://aws.amazon.com](http://aws.amazon.com)
- Cloudera: [http://www.rackspace.co.uk/cloud](http://www.rackspace.co.uk/cloud)
- Google Cloud Platform: [https://cloud.google.com](https://cloud.google.com)
- Microsoft Azure: [http://azure.microsoft.com](http://azure.microsoft.com)

### Create a key pair

In this exercise we will be working with Amazon Web Services. 

Amazon uses public–key cryptography to authenticate users, so we'll need to create a private/public key pair:

``` bash
# Create a key pair
$ ssh-keygen -t rsa -f ~/.ssh/ec2 -b 4096
```

We then need to register our public key with Amazon Web Services.

![](session10/figures/key_pair.png)

### Create a single instance using the web interface

Navigate to the EC2 Dashboard and create a 'micro' instance (1 CPU, 2.5GHZ, 1GB RAM):

![](session10/figures/create_ec2_instance.png)

### Connect to the instance with SSH

For Linux instances, the username is 'ec2-user':

``` bash
# <key_file>: ~/.ssh/ec2
$ ssh ec2-user@52.16.96.114 -i <key_file> 
```

![](session10/figures/connect_to_instance.png)

``` bash
# Connected 
       __|  __|_  )
       _| \(     /   Amazon Linux AMI
      ___|\\___|___|

[ec2-user@ip-172-31-5-39 ~]$ 
```

### Upload and run a script from your local machine

``` bash
# Add a hello world here

# transfer the file from your local machine
$ $ scp -i <key_file> <SampleFile> ec2-user@ec2-198-51-100-1.compute-1.amazonaws.com:~
```

### Terminate the server

When finished with the instance, remember to terminate it!

![](session10/figures/terminate_instance.png)
