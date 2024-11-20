function NetworkEfficiencyLoop(PlatePos, Plate, CrossDate, HarvestDate)

%Create list of time points (directories with images of the required sample)
dd=dir(['*_Plate' PlatePos]);
ind = [];
for i = 1:length(dd)
    ns = dd(i).name;
    if str2double(ns(1:8))>=CrossDate && str2double(ns(1:8))<=HarvestDate
        ind = [ind,i];
    end
end
dd = dd(ind);
clear i ind ns

%Empty variables to store efficiency (global and root) and cost for the
%different networks (MST, DT & experiments)
Er = []; ErMST = []; ErDT = [];
Eg = []; EgMST = []; EgDT = [];
C = []; CMST=[]; CDT = [];
timeStep = []; time = [];

for i =1:length(dd)
    ns = dd(i).name;
    cd(ns)
    [distGlobal , distRoot, distGlobalMST , distRootMST, sPathGlobal , sPathRoot, sPathGlobalMST , sPathRootMST, sPathGlobalDT , sPathRootDT, Cost, CostMST, CostDT] = NetworkEfficiency;
    
    if isempty(Cost)==0
        
        Er       = [Er, sum(distRoot./sPathRoot)/length(sPathRoot)];
        Eg       = [Eg, sum(distGlobal./sPathGlobal)/length(sPathGlobal)];
        ErMST    = [ErMST, sum(distRootMST./sPathRootMST)/length(sPathRootMST)];
        EgMST    = [EgMST, sum(distGlobalMST./sPathGlobalMST)/length(sPathGlobalMST)];
        ErDT     = [ErDT, sum(distRoot./sPathRootDT)/length(sPathRootDT)];
        EgDT     = [EgDT, sum(distGlobal./sPathGlobalDT)/length(sPathGlobalDT)];
                
        C        = [C,    Cost];
        CMST     = [CMST, CostMST];
        CDT      = [CDT,  CostDT];
        timeStep = [timeStep, i];
        time     = [time; ns(1:13)];
    end
    cd ..
end

save(['Analysis_001P100N\Plate' sprintf('%03.0f',Plate) '\Efficiency.mat'],...
    'Er','Eg','ErMST','EgMST','ErDT','EgDT', ...
    'C', 'CMST', 'CDT' ,'timeStep','time')
