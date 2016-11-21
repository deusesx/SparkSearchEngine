import os
import re
import shutil
from operator import add
from pyspark.sql.types import *
from pyspark.sql.functions import *


class SearchEngine():
    """
    Search Engine class used for constructing inverted index, process queries and manipulate with index files.
    Author: Artur Samigullin
    """
    
    def __init__(self, sc, sqlc):
        """
        Constructor

        Initializes SearchEngine class with empty index, passed SparkContext and SQLContext

        Parameters
        ----------
        sc : SparkContext
            An initialized SparkContext
        sqlc : SQLContext
            An initialized Spark SQLContext
        """
        self.index = None
        self.sc = sc
        self.sqlc = sqlc
    
    
    def construct_index(self, input_path):
        """
        Indexing function.

        Function to index a corpus of preprocessed text files.

        Parameters
        ----------
        input_path : string
            Should be path to existing folder
        """
        rdd = self.sc.wholeTextFiles(input_path)\
        .flatMap(lambda name_content: map(lambda word: (word, name_content[0]), name_content[1].split()))\
        .map(lambda word_name: ((word_name[0], word_name[1]), 1))\
        .reduceByKey(lambda count1, count2: count1 + count2)\
        .map(lambda word_name_count: (word_name_count[0][0], word_name_count[0][1], word_name_count[1]))
        
        fields = [StructField("Word", StringType(), False), StructField("Name", StringType(), False), StructField("Count", IntegerType(), False)]
        self.index = self.sqlc.createDataFrame(rdd, StructType(fields))
        self.index.createOrReplaceTempView("index_view")
    
    
    def search(self, query_string):
        """
        Searching function.

        Function for searching on built index. Query string must follow same preprocessing steps as corpus.

        Parameters
        ----------
        query_string : string
            A string in format 'Token0 Token1 ... TokenN'

        Returns
        -------
        Spark DataFrame (ResilentDistributedDataset)
            Returns search result, sorted by number of hits
        """
        search_condition = " OR ".join(map(lambda keyword: "word = '%s'" % keyword, query_string.split()))
        result_rdd = self.index.select("*")\
        .filter(search_condition)\
        .rdd.map(lambda row: (row.Name, row.Count))\
        .reduceByKey(lambda count1, count2: count1 + count2)\
        .sortBy((lambda name_count: name_count[1]), False)\
        .map(lambda name_count: name_count[0] + " has number of hits: " + str(name_count[1]))
        return result_rdd
        
        
    def save_index(self, output_filename):
        """
        Save index on disk.

        Function for saving index in 'Parquet' formatted file.
        Note: if file is already exists, it will be overwritten.
        
        Parameters
        ----------
        output_filename: string
           Path for saving index in 'Parquet' formatted file.
        """
        if os.path.exists(output_filename):
            if os.path.isfile(output_filename):
                os.remove(output_filename)
            else:
                shutil.rmtree(output_filename)
        self.index.write.parquet(output_filename)

        
    def load_index(self, input_filename):
        """
        Load index from disk.

        Function for loading index from 'Parquet' formatted file.

        Parameters
        ----------
        input_filename : string
            Path for loading index from 'Parquet' formatted file.
        """
        self.index = self.sqlc.read.parquet(input_filename)
        self.index.createOrReplaceTempView("index_view")
