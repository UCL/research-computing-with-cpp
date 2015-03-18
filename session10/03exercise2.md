---
title: Exercise 2 - Working in the cloud
---

## Exercise 2: Working in the cloud

### Again, from the command line...

Install Amazon Web Services Command Line Interface:  
[http://docs.aws.amazon.com/cli/latest/userguide/installing.html](http://docs.aws.amazon.com/cli/latest/userguide/installing.html)

``` bash
# install the tools with pip
sudo pip install awscli
aws ec2 help
```

### Configure the tools

To use the command line tools, you'll need to configure your AWS Access Keys, region, and output format:  
[http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

``` bash
$ aws configure

# This configuration is stored locally in a home folder named .aws
# On Unix systems: ~/.aws/config; 
# On Windows: %UserProfile%\.aws\config
AWS Access Key ID [****************VDLA]: EXXXXXAMPLE
AWS Secret Access Key [****************pa8o]: EXXXXXAMPLE
Default region name [eu-west-1]: eu-west-1
Default output format [json]: json
```

### Test our connection to Amazon Web Services

If our connection has been set up correctly, 'describe-regions' will return a list of Amazon Web Service regions:

``` bash
# Successful connection will return list of AWS regions
# HTTPSConnectionPool error? Try changing region to eu-west-1
$ aws ec2 describe-regions
```

### Create a key pair

To connect to the instance, we will need a key pair. If you haven't already done so, create one now:

``` bash
# Create an SSH key pair
$ ssh-keygen -t rsa -f ~/.ssh/ec2 -b 4096
```

Transfer the public key to AWS:

``` bash
# <key_name> is a unique name for the pair (e.g. my-key)
# <key_blob> is the public key: "$(cat ~/.ssh/ec2.pub)"
$ aws ec2 import-key-pair --key-name <key_name> \
  --public-key-material <key_blob>
```

### Create a security group

We'll also need to create a security group...

``` bash
# creates security group named my-security-group
$ aws ec2 create-security-group \
  --group-name "My security group" \
  --description "SSH access from my local IP address"
```

### Configure the security group

...and allow inbound connections from our local IP address:

``` bash
# create a rule to allow inbound connections on TCP port 22
# find your IP: curl http://checkip.amazonaws.com/
$ aws ec2 authorize-security-group-ingress \
  --group-name "My security group" \
  --cidr <local_IP_address>/32 \
  --port 22 \
  --protocol tcp
```

Note: the /32 at the end of the IP address is the bit number of the [CIDR netmask](http://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing). The /32 mask is equivalent to 255.255.255.255, so defines a single host. Lower values broaden the range of allowed addresses. An IP of 0.0.0.0/0 would allow all inbound connections.

### Locate an appropriate Machine Image

An Amazon Machine Image contains the software configuration (operating system, software...) needed to launch an instance. AMIs are provided by:

- Amazon Web Services
- the user community
- AWS Marketplace

We will search for an Amazon Machine Image ID (AMI-ID) using the command line tools:

``` bash
# Use filter to locate a specific machine (ami-9d23aeea)
# Filter is a key,value pair
$ aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=amzn-ami-hvm-2014.09.2.x86_64-ebs"
```

### Launch an instance

Launch an instance using the Amazon Machine Image ID:

``` bash
# <AMI-ID>: ami-9d23aeea
# <key_name>: defined when transferring the key
# <group_name>: "My security group"
$ aws ec2 run-instances --image-id <AMI-ID> \
  --key-name <key_name> \
  --instance-type t2.micro \
  --security-groups <group_name>
```

### View the instance

We can check the instance and find the public IP:

``` bash
# View information about the EC2 instances
# e.g. state, root volume, IP address, public DNS name
$ aws ec2 describe-instances
```

### Connect to the instance

Use the public ID to connect:

``` bash
# for Linux instances, the username is ec2-user
# <public_ID>: 52.16.106.209
# <key_file>: ~/.ssh/ec2
$ ssh -i <key_file> ec2-user@<public_ID>
```

You should now be connected!:

``` bash
# Connected 
       __|  __|_  )
       _| \(     /   Amazon Linux AMI
      ___|\\___|___|

[ec2-user@ip-172-31-5-39 ~]$ 
```

<!-- ### Download and run a script from the web

``` bash
# wget a file from the web (boids?)
[ec2-user@ip-172-31-5-39 ~]$ wget <url>
```
 -->

### Terminate the instance

Don't forget to terminate the instance when you have finished:

``` bash
# terminate the instance 
# <InstanceId>: i-87086760
$ aws ec2 terminate-instances --instance-ids <InstanceId>

TERMINATINGINSTANCES    i-87086760
CURRENTSTATE    32  shutting-down
PREVIOUSSTATE   16  running
```

