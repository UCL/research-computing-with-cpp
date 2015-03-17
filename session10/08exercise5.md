---
title: Exercise 5 - Distributed computing
---

## Exercise 5: Distributed computing

### Again, with Hadoop

In the previous exercise, we demonstrated MapReduce on our local computer.

In this exercise we will spin up multiple cloud instances, making use of Hadoop to carry out the MapReduce operation.

We could set up multiple instances in the cloud (for example with EC2), then configure Hadoop to run across the instances. This is not trivial and takes time. 

### Hadoop in the cloud

Fortunately, several cloud providers offer configurable Hadoop services. One such service is Amazon Elastic MapReduce (EMR).

Elastic MapReduce provides a framework for:

- uploading data and code to the Simple Storage Service (S3)
- analysing with a multi-instance cluster on the Elastic Compute Cloud (EC2)

### Create an S3 bucket

Create an S3 bucket to hold the input data and our map/reduce functions:

1. Open the Amazon Web Services Console: http://aws.amazon.com/
2. Select "Create Bucket" and enter a globally unique name
3. Ensure the S3 Bucket shares the same region as other instances in your cluster

Or, through the CLI:
```
aws s3 mb s3://ucl-jh-books-example
```

### Copy data and code to S3

The sample map and reduce functions are available on GitHub:

``` bash
# clone the code from a remote repository
$ git clone https://github.com/tompollard/dorian
```

Copy the data and code to S3:

``` bash
# Copy input code and data to S3
# No support for unix-style wildcards
$ aws s3 cp dorian.txt s3://my-bucket-ucl123/input/
$ aws s3 cp mapper.py s3://my-bucket-ucl123/code/
$ aws s3 cp reducer.py s3://my-bucket-ucl123/code/
```

### Launch the compute cluster

Create a compute cluster with one master instance and two core instances:

``` bash
# Start a EMR cluster
# <ami-version>: version of the machine image to use
# <instance-type>:  number and type of Amazon EC2 instances
# <key_name>: "mykeypair"
$ aws emr create-cluster --ami-version 3.1.0 \
    --ec2-attributes KeyName=<key_name> \
    --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m3.xlarge \
    InstanceGroupType=CORE,InstanceCount=2,InstanceType=m3.xlarge

"ClusterId": "j-3HGKJHEND0DX8"
```

### Get the cluster ID

Get the cluster-id:

``` bash
# Get the cluster-id
$ aws emr list-clusters
    ...
    "Id": "j-3HGKJHEND0DX8", 
    "Name": "Development Cluster"
```

### Get the public DNS name of the cluster

When the cluster is up and running, get the public DNS name:

``` bash
# Get the DNS
$ aws emr describe-cluster --cluster-id j-3HGKJHEND0DX8 \
    | grep MasterPublicDnsName
    ...
    "MasterPublicDnsName": "ec2-52-16-235-144.eu-west-1.compute.amazonaws.com"
```

### Connect to the cluster

SSH into the cluster using the username 'hadoop':

``` bash
# SSH into the master node
# <key_file>: ~/.ssh/ec2
# <MasterPublicDnsName>: ec2-52-16-235-144.eu-west-1.compute.amazonaws.com
$ ssh hadoop@<MasterPublicDnsName> -i <key_file> 
```

``` bash
# Connected 
       __|  __|_  )
       _| \(     /   Amazon Linux AMI
      ___|\\___|___|

[hadoop@ip-x ~]$ 
```

### Run the analysis

``` bash
# To process multiple input files, use a wildcard
[hadoop@ip-x ~]$ hadoop \
    jar contrib/streaming/hadoop-*streaming*.jar \
    -files s3://my-bucket-ucl123/code/mapper.py,s3://my-bucket-ucl123/code/reducer.py \
    -input s3://my-bucket-ucl123/input/* \
    -output s3://my-bucket-ucl123/output/ \
    -mapper mapper.py \
    -reducer reducer.py
```

### View the results

Results are saved to the output folder. Each reduce task writes its output to a separate file:

``` bash
# List files in the output folder
$ aws s3 ls s3://my-bucket-ucl123/output/
```

Download the output:

``` bash
# Copy the output files to our local folder
# No support for unix-style wildcards, so use --recursive
$ aws s3 cp s3://my-bucket-ucl123/output . --recursive

# View the file
$ head part-00001

the     3948
and     2279
in      1266
his     996
lord    248
...
```

### Terminate the cluster

Once our analysis is complete, terminate the cluster:

``` bash
# get cluster id: aws emr list-clusters
# <cluster_ID>: j-3HGKJHEND0DX8
$ aws emr terminate-clusters --cluster-id <cluster_ID>
```

### Delete the bucket

``` bash
aws s3 rb s3://my-bucket-ucl123 --force
```
