Short description:
1) BASrate.m labels every edge, from the network graph, as runner hypha (RH) or branched absorbing structures (BAS). 
It saves (appends) this information inside the already existing network graph file.

Instructions:
1) This code uses the network graph extracted from the raw images. The graph must be saved in "Analysis\graph_full_labeled_masked.m" inside the directory of a single time step (yyyymmdd_hhMM_Platexx, where xx is the holder position in the setup). 
2) Run BASrate.m in the folder path containing all time steps. The input parameters are PlatePos, Holder position (string); Plate, sample number (number); CrossDate, crossing date (number) in the format yyyymmdd; HarvestDate, end date of the experiment (number) in the format yyyymmdd.
3) The labeling of RH or BAS is appended inside the existing network graph "Analysis\graph_full_labeled_masked.m"
4) For all time steps a summarized table is saved inside "Analysis_001P100N\Platexxx\BASrate.mat", where xxx is plate/sample number. It contains #BAS edges, #RH edges, Total length of BAS, Total length of RH, real time, and time step.
5) A colored image of the network skeleton, distinguishing BAS from RH in each time step, is saved in "Analysis_001P100N\Platexxx\SkelBAS" for validation purposes.