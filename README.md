## Data Engineer Assignment

This assignment consists of two parts, an architectural design and a coding challenge. 

# Architecture

We have many internal systems generating events at a peak rate of ~100k events/s (e.g market 
data updates, prices and trades). These events are highly structured and include identifiers 
that relate to events and metadata in other systems. The source systems do not support 
back-pressure, any dropped messages are lost.

Design a system that is able to process and enrich these events with information from other systems,
and then display a dynamic selection of the information to multiple users in real-time with updates. 
We expect a system level design describing the major components and reasons for using them. 
Limit the detail to a single page.

# Spark

Attached is a small PySpark project. There are two datasets, trades and prices, that need to 
be combined in two ways. Examples of the expected output are in the source.

Your answer should be a runnable `main.py` script that shows the results from each join operation,
matching the example in the docstring.

Include a discussion of your solution, and how would scale to 100,000s events over multiple days 
and 100s of ids, and if your approach would be different at that scale.
