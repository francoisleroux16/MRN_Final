# MRN_Final
Full Code used for the MRN project

## How it works
A brief summary of each of the programs making up the project code will be given below, specifically on how to use it. To read more about how the program was developed, where the processes and equations come from, please refer to the report: "Visualize the Flow Field behind an Aircraft", written by Francois le Roux at the University of Pretoria (up.ac.za) - the Report may only become available to the public in 2021.

### importing.py

This program handles all of the importing used within the other programs. It is only required to be imported into the other programs and used. It provides a handy GUI for the user to select the data they wish to use.

### calibration.py

This program is used for the calibration of the five-hole probe. There are two scenarios to use the program, the first is to convert the raw data into a spreadsheet of calibration coefficients, and the second is to plot all of the calibration plots using a spreadsheet of calibration coefficients.

The raw data can be imported using the importing.py program. Use the 'generate_calibration_data' function with the imported raw data as input to the function to generate the calibration coefficients spreadsheet. The user will be asked to provide a name for the spreadsheet as input.

All of the plots can be generated using the 'plot_all' function. The function takes care of everything, and prompts the user to select the spreadsheet containing the calibration coefficients. 10 plots will be generated for the user.

### testdata.py

This program changes the experimental pressure readings obtained by the five-hole probe into velocity vectors. Due to the complexity and size of the program functions were generated to complete multiple steps for the user. 

The function 'do_test' imports all of the necessary data for the user: The experiment data, the experiment position data (received as war output from LabVIEW) and the calibration coefficients generated in 'calibration.py'. Next the user has to merge the two experimental datasheets: first obtain the position data in the right shape using 'get_position_data', and then merge it with the experiment data using 'merge_data'. The reason this is not automated is because the user can have custom shapes being used by the probe, and a pattern will not neccessary be available for the program to automate the process.

Once the experimental data has been merged, use the 'make_big_array' function to generate an array containing all of the available data at each measurement point. Finally, use a 'get_Velocity' function to generate the velocity data from the array, and apply the function 'downwash_correction' on the function. Your velocity data is now generated for each sample point. Use 'basic_quiver' to plot the velocity data.

#### NOTE
Should the data be in the same shape as the data used in the experiment, i.e. 30x30 - simply run the single function 'run_all' - this will run everything and present the resulting quiver plot

