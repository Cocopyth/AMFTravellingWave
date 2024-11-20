function BASrate(PlatePos, Plate, CrossDate, HarvestDate)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%  input: Plate parameters (position in setup, plate number, crossing date,
%         and harvest date (end of experiement)
%  output: RBAS
%          col1: # BAS, col2: # RH, col3: length BAS, col4: length RH.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%

%Create the list of images in dd.
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

%Create folder to save output RBAS and colored images of the skeleton
%distinguishing BAS out of RH.
mkdir(['Analysis_001P100N/Plate' sprintf('%03.0f',Plate) '/SkelBAS'])

%BAS extraction starts here
RBAS = [];
timeStep = [];
time     = [];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
figure

for q = 1:length(dd)
    
    ns = dd(q).name;
    
    if exist([ns '/Analysis/graph_full_labeled_masked.mat'])
        %Uses network graph information to discriminate BAS from RH
        disp (['Extracting BAS ratio on ' ns])
        load([ns '/Analysis/graph_full_labeled_masked.mat'])
        if iscell(width)
            for i = 1:length(width)
                w(i) = double(width{i});
            end
            width = w; 
            %Width of every edge in the network is saved in this array
        end
        
        %ind is the index of elements inside the ROI
        ind = find(iEnd+iOrigin==2);
        
        
        %Finding end points in the list of nodes (ind)
        iendPoint = [];
        for i = ind
            if isempty(find(origin_pos_masked(i) == end_pos_masked,1))==1 && length(find(origin_pos_masked(i) == origin_pos_masked))==1
                iendPoint=[iendPoint,i];
            end
            if isempty(find(end_pos_masked(i) == origin_pos_masked,1))==1 && length(find(end_pos_masked(i) == end_pos_masked))==1
                iendPoint=[iendPoint,i];
            end
        end
        
        %Loop to extract local number of nodes, and hypha length and width.
        R = 250;
        L = [];
        N = [];
        W = [];
        k = 1;
        for i = ind
            x = mean([origin_pos_masked(i,1) end_pos_masked(i,1)]);
            y = mean([origin_pos_masked(i,2) end_pos_masked(i,2)]);
            nOrigin = find(origin_pos_masked(:,1)>x-R & origin_pos_masked(:,1)<x+R & ...
                origin_pos_masked(:,2)>y-R & origin_pos_masked(:,2)<y+R);
            nEnd    = find(end_pos_masked(:,1)>x-R    & end_pos_masked(:,1)<x+R & ...
                end_pos_masked(:,2)>y-R    & end_pos_masked(:,2)<y+R);
            nRep = 0;
            for j =1:length(nOrigin)
                nRep = nRep + length(find(origin_pos_masked(nOrigin(j)) == end_pos_masked(nEnd)));
            end
            N(k) = length(nOrigin) + length(nEnd) - nRep;
            L(k) = length(pixel_list_masked{i});
            W(k) = width(i);
            k    = k+1;
        end
        
        
        k = 1; l = 1;
        indBAS = []; lengthBAS=[]; indRH=[]; lengthRH=[];
        for i = 1:length(ind)
            hypha_masked = pixel_list_masked{ind(i)};
            if W(i)*L(i)<3000 && W(i)<7
                plot(hypha_masked(:,2),16000-hypha_masked(:,1),'color',[185/255 67/255 151/255])
                indBAS(k)=ind(i);
                lengthBAS(k)=length(hypha_masked);
                k = k+1;
            elseif L(i)<200
                plot(hypha_masked(:,2),16000-hypha_masked(:,1),'color',[185/255 67/255 151/255])
                indBAS(k)=ind(i);
                lengthBAS(k)=length(hypha_masked);
                k = k+1;
            elseif length(find(ind(i)==iendPoint))==1 && L(i)<500
                plot(hypha_masked(:,2),16000-hypha_masked(:,1),'color',[185/255 67/255 151/255])
                indBAS(k)=ind(i);
                lengthBAS(k)=length(hypha_masked);
                k = k+1;
            else
                plot(hypha_masked(:,2),16000-hypha_masked(:,1),'color',[119/255 172/255 50/255])
                indRH(l)=ind(i);
                lengthRH(l)=length(hypha_masked);
                l = l+1;
            end
            hold on
        end
        axis equal
        ylim([0 16000]), xlim([0 40000])
        saveas(gcf,['Analysis_001P100N/Plate' sprintf('%03.0f',Plate) '/SkelBAS/' ns '.tiff'])
        RateBAS = [length(indBAS) length(indRH) sum(lengthBAS) sum(lengthRH)];
        save([ns '/Analysis/graph_full_labeled_masked.mat'],'L','W','N','iendPoint','indBAS','indRH','RateBAS','-append')
        
        RBAS = [RBAS ; RateBAS];
        
        timeStep = [timeStep, i];
        time     = [time; ns(1:13)];
    end
    
end

save(['Analysis_001P100N\Plate' sprintf('%03.0f',Plate) '\BASrate.mat'], 'RBAS', 'time', 'timeStep')

