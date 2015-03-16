---
title: Exercise 5 - Distributed computing
---

## Exercise 5: Distributed computing

### Again, with Hadoop

In the previous exercise, we demonstrated the Map/Reduce functions on our local computer.

In this exercise we will spin up multiple cloud instances, making use of Hadoop to carry out a Map Reduce operation.

We could set up multiple instances in the cloud (for example with EC2), then configure Hadoop to run across the instances. This is not trivial and takes time. 

### Hadoop in the cloud

Fortunately, several cloud providers offer configurable Hadoop services. One such service is Amazon Elastic MapReduce.

Elastic MapReduce (EMR) provides a framework for:

- uploading data and code to the Simple Storage Service (S3)
- analysing with a multi-instance cluster on the Elastic Compute Cloud (EC2)

### Create an S3 bucket

Create an S3 bucket to hold the: 

- input (the book and our map/reduce functions)
- logs
- output (results from our analysis)

1. Open the Amazon Web Services Console: http://aws.amazon.com/
2. Select "Create Bucket" and enter a globally unique name
3. Ensure the S3 Bucket shares the same region as other instances in your cluster
4. Create subfolders for the input, logs, and output (e.g. ```s3://my-bucket-ucl123/output```)

<!-- 
May need to do more from here: http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-cli-install.html
-->

### Launch the compute cluster

Create a compute cluster with one master instance and two core instances:

``` bash
# Start a EMR cluster
# <ami-version>: version of the machine image to use
# <instance-type>:  number and type of Amazon  EC2  instances
$ aws emr create-cluster --ami-version 3.1.0  \
    --auto-terminate \
    --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m3.xlarge InstanceGroupType=CORE,InstanceCount=2,InstanceType=m3.xlarge
```

list-clusters
list-instances
terminate-clusters

### Connect to the cluster and run the analysis

``` bash
hadoop \
   jar /opt/hadoop/contrib/streaming/hadoop-streaming-1.0.3.jar \
   -mapper "python $PWD/mapper.py" \
   -reducer "python $PWD/reducer.py" \
   -input "wordcount/mobydick.txt"   \
   -output "wordcount/output"
```

### View the results

Results can be viewed in the output folder:

``` bash
"results"
```


