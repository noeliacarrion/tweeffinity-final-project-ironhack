# Tweeffinity

This is the final project of Ironhack data analysis bootcamp. Tweeffinity is a program that discovers the affinity between two twitter users. Through the HDBSCAN cluster model and with natural language processing techniques the program returns an affinity percentage based on shared content.

The project has the following folders.

### - src

It contains all the files to run the program. It is only necessary to run the main.py file in the terminal by entering two twitter users. Here, there is an example: 

<img width="260" alt="Captura de pantalla 2019-08-13 a las 15 21 36" src="https://user-images.githubusercontent.com/49640612/63044921-8d524e80-becf-11e9-9568-a3dd5d196648.png">

### - jupyter notebook

It contains the different tests that have been done to test the best machine learning model of clustering.

### - input

It contains 2 csv files with all tweets from both users.

### - output

It contains the result after running the program:

- Two wordclouds from user1 and user2 with the most used words 
- An image with all the tweets from user1 grouped by the the type of content.
- An image with the comparison between both users.

