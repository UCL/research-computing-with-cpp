---
title: Distributed computing
---

## Distributed computing

### Distributed computing

Dividing a problem into many tasks means analysis can be shared across multiple computers, in a parallel fashion.

Cloud computing systems have made distributed computing increasingly accessible to individual users.

Introduced by Google in 2004, MapReduce is a popular model that supports distributed computing on large data sets across clusters of computers.

### MapReduce

Content to follow.

<!-- 
Processing occurs in two basic steps:

Map step: master node takes the input, partitions it into smaller, sub-problems, and distributes those to worker notes. A worker node may do this again, leading to a tree structure. The worker node processes the smaller problem and passes the answer back to its master node.

Reduce step: master node takes the answers to all of the sub problems and combines them in some way to get the output - the answer to the question it was originally trying to solve.

-->

### Distributed file systems

Content to follow.

<!-- 
Distributed file systems have developed to provide reliable data storage across a distributed system.
-->

### Hadoop

Content to follow.

<!-- 
http://hadoop.apache.org/docs/r1.2.1/mapred_tutorial.html#Reducer

Apache Hadoop is an implementation of MapReduce. Amazon Elastic MapReduce is an implementation of Hadoop MapReduce on the Amazon Web Services.

Combines map reduce with HDFS. Can be run across multiple servers. It is possible to use Hadoop across multiple servers. Requires fairly involved configuration process. Installing Hadoop, opening firewalls etc. 

Nodes, cluster etc. Need to explain Master and Core Instance. What are bootstrap actions?

Fortunately services offer pre-configured clusters. We will use Amazon Elastic MapReduce...

Hive is a tool for managing data on a distributed encironment and provides an SQL-like qery language.

Using "streaming" any language can be used to implement the map and reduce.
-->

