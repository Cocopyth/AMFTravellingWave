Short description:
1) Networkefficiency.m extracts the necessary information in order to calculate the efficiency. This is: minimum distance between nodes (dist), actual distance between nodes (network paths, sPath) and cost. For the original network graph and the 2 ideal ones (MST and DT). 
It saves this information inside the "Analysis\NetworkEfficiency.mat" directory inside the original one containing the images.
2) NetworkEfficiencyLoop.m uses the information "Analysis\NetworkEfficiency.mat" and calculates the efficiency for evy time step in a single sample.
It saves this information in an external directory "Analysis_001P100N\Platexxx", where xxx is the sample number.

Instructions:
1) This code uses the network graph extracted from the raw images. The graph must be saved in "Analysis\graph_full_labeled_masked" inside the directory of a single time step (yyyymmdd_hhMM_Platexx, where xx is the holder position in the setup). 
2) Run NetworkEfficiencyLoop.m in the folder path containing all time steps. The input parameters are PlatePos, Holder position (string); Plate, sample number (number); CrossDate, crossing date (number) in the format yyyymmdd; HarvestDate, end date of the experiment (number) in the format yyyymmdd.
3) The calculation of the efficiencies, root and global, for the 3 network graphs in each time step is saved inside "Analysis_001P100N\Platexxx\Efficiency.mat", where xxx is plate/sample number.