---
title: Distributed computing
---

## Distributed computing

### Distributed computing

Dividing a problem into many tasks means analysis can be shared across multiple computers, in a parallel fashion.

Cloud computing systems have made distributed computing increasingly accessible to individual users.

Introduced by Google in 2004, MapReduce is a popular model that supports distributed computing on large data sets across clusters of computers.

### MapReduce

MapReduce systems are built around the concepts of:

- a mapper, in which the master node takes an input, separates it into sub-problems, and distributes those to worker notes
- a 'shuffle' step to distribute data from the mapper
- a reducer which collates the answers to the sub-problems and combines them to solves the initial question.

### Distributed file systems

Distributed file systems

- developed to provide reliable data storage across distributed systems
- replicates data across multiple hosts to achieve reliability

### Hadoop

Apache Hadoop:

- framework for distributed processing of large data sets 
- scales from single servers to many machines
- includes Hadoop MapReduce and the Hadoop Distributed File System

<!-- 
http://hadoop.apache.org/docs/r1.2.1/mapred_tutorial.html#Reducer

Apache Hadoop is an implementation of MapReduce. Amazon Elastic MapReduce is an implementation of Hadoop MapReduce on the Amazon Web Services.

Combines map reduce with HDFS. Can be run across multiple servers. It is possible to use Hadoop across multiple servers. Requires fairly involved configuration process. Installing Hadoop, opening firewalls etc. 

Nodes, cluster etc. Need to explain Master and Core Instance. What are bootstrap actions?

Fortunately services offer pre-configured clusters. We will use Amazon Elastic MapReduce...

Hive is a tool for managing data on a distributed encironment and provides an SQL-like qery language.

Using "streaming" any language can be used to implement the map and reduce.
-->

