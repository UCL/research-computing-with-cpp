---
title: Distributed computing
---

## Distributed computing

### Distributed computing

Dividing a problem into many tasks means analysis can be shared across multiple computers, in a parallel fashion.

Cloud computing systems have made distributed computing increasingly accessible to individual users.

Introduced by Google in 2004, MapReduce is a popular model that supports distributed computing on large data sets across clusters of computers.

### MapReduce

MapReduce systems are built around the concepts of:

- a map method, in which the master node takes an input, separates it into sub-problems, and distributes those to worker notes
- a reduce method which collates the answers to the sub-problems and combines them to solves the initial question.

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

