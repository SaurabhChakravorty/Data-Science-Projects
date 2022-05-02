# blue-avenir

The following 'case study is solved using FAST API as a framework and POSTGRE SQL is used as a database. The whole application is containerized using Docker. Swagger is provided for the end points in the local host address.

## File configuration
Following are the file configuration specifications:
1. 'app.py': This file contains the endpoints of the server and by running this file we get our web page with GET, POST and PUT request methods in Swagger.
2. 'create_database.py': This file needs to be run initially to create tables. The file is synced with the POSTGRE SQL engine. So, make sure you run the server first by creating a local database named 'Blue_Avenir'
3. 'batch_process.py': It contains all the modules for preprocessing the data after we get the data from the user.
4. 'RL_output.py': This is the RL function that gives the prediction after the states and offers are entered in it.
5. 'run.py': It contains all the rendered data format for taking input from the user.

## Steps to run
The files can be run locally in a suitable IDE ideally PyCharm or by creating a docker container. We have used a local database so make sure to establish a database first locally. The steps to run these files are:
1. Run the 'create_databse.py' file to initialize the database with proper schema.
2. Run 'app.py' to get the server running at the localhost address.
3. The data can be uploaded at respective endpoints to get the response from the server from the files.
NOTE: The database entry will automatically be created for all the response queries from the server as the scripts are designed to automate the whole process.
