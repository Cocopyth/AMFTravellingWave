function [distGlobal , distRoot, distGlobalMST , distRootMST, sPathGlobal , sPathRoot, sPathGlobalMST , sPathRootMST, sPathGlobalDT , sPathRootDT, Cost, CostMST, CostDT] = NetworkEfficiency

conv_px2mum = 1.725; %COnversion pixels to micrometers in the setup

load('Analysis\graph_full_labeled_masked.mat')
origin_label = double(cell2mat(origin_label));
end_label    = double(cell2mat(end_label));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% posNodes_ALL contains the position of all nodes in ROI (including BAS)
% centers_ALL contains the label of all nodes in ROI (including BAS)
ind          = find(iEnd+iOrigin==2);
oL           = origin_label(ind)'+1;
eL           = end_label(ind)'+1;
ns           = [oL;eL];
[cs,centers] = hist(ns,0:max(ns));
centers      = centers(cs>0);
Nodes        = [origin_pos_masked(ind,:);end_pos_masked(ind,:)];
posNodes     = [];
for k = 1:length(centers)
    posNodes(k,:)= Nodes(find(ns==centers(k),1),:);
end


sPathGlobal = []; sPathGlobalMST = []; sPathGlobalDT = [];
distGlobal  = []; distGlobalMST  = [];
sPathRoot = []; sPathRootMST = []; sPathRootDT = [];
distRoot  = []; distRootMST  = [];

if length(posNodes)>=10 %Only calculates efficiency in a network conaining more than 10 nodes.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Creating Graphs
    %Original graph
    G    = graph(oL,eL,length_mum(ind));
    Cost = sum(G.Edges.Weight);
    %Minimum Spanning Tree graph
    Gmst = minspantree(G,'Method', 'sparse','Root',min([oL;eL]));
    CostMST = sum(Gmst.Edges.Weight);
    %Delaunay Triangulation graph
    DT   = delaunayTriangulation(posNodes);
    conn = DT.edges;
    dist = sqrt((posNodes(conn(:,1),1)-posNodes(conn(:,2),1)).^2+(posNodes(conn(:,1),2)-posNodes(conn(:,2),2)).^2)*conv_px2mum;Gdt  = graph(centers(conn(:,1)),centers(conn(:,2)),dist);
    Gdt  = graph(centers(conn(:,1)),centers(conn(:,2)),dist);
    CostDT = sum(Gdt.Edges.Weight);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % posNodes contains the position of RH nodes (excluding BAS)
    % centers contains the label of RH nodes (excluding BAS)
    % Nodes are divided in Root or Tip, depending of it's position (barrier)
    oL_RH             = origin_label(indRH)'+1;
    eL_RH             = end_label(indRH)'+1;
    origin_pos_masked = origin_pos_masked(indRH,:);
    end_pos_masked    = end_pos_masked(indRH,:);
    
    ns           = [oL_RH;eL_RH];
    [cs,centers] = hist(ns,0:max(ns));
    centers      = centers(cs>0);
    Nodes        = [origin_pos_masked;end_pos_masked];
    posNodes     = [];
    for k = 1:length(centers)
        posNodes(k,:)= Nodes(find(ns==centers(k),1),:);
    end
    
    nsRoot            = [oL_RH(origin_pos_masked(:,1)>14500);...
        eL_RH(end_pos_masked(:,1)>14500)];
    [cs,centersRoot]  = hist(nsRoot,0:max(nsRoot));
    centersRoot       = centersRoot(cs>0);
    NodesRoot         = [origin_pos_masked(origin_pos_masked(:,1)>14500,:);...
        end_pos_masked(end_pos_masked(:,1)>14500,:)];
    posNodesRoot      = [];
    for k = 1:length(centersRoot)
        posNodesRoot(k,:) = NodesRoot(find(nsRoot==centersRoot(k),1),:);
    end
    
    nsTip             = [oL_RH(origin_pos_masked(:,1)<=14500);...
        eL_RH(end_pos_masked(:,1)<=14500)];
    [cs,centersTip]   = hist(nsTip,0:max(nsTip));
    centersTip        = centersTip(cs>0);
    NodesTip          = [origin_pos_masked(origin_pos_masked(:,1)<=14500,:);...
        end_pos_masked(end_pos_masked(:,1)<=14500,:)];
    posNodesTip       = [];
    for k = 1:length(centersTip)
        posNodesTip(k,:) = NodesTip(find(nsTip==centersTip(k),1),:);
    end
    
    
    %% Global efficiency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ind = 2;
    for i = 1:length(centers)-1
        for j = ind:length(centers)
            [~,d]          = shortestpath(G,centers(i),centers(j));
            if isinf(d) == 0 && d >= 1
                sPathGlobal    = [sPathGlobal ; d];
                distNodes      = round(norm(posNodes(i,:)-posNodes(j,:))*conv_px2mum);
                
                [~,d]          = shortestpath(Gdt,centers(i),centers(j));
                sPathGlobalDT  = [sPathGlobalDT ; d];
                
                [~,dm]          = shortestpath(Gmst,centers(i),centers(j));
                if isinf(dm) == 0  && dm >= 1
                    sPathGlobalMST = [sPathGlobalMST ; dm];
                    distGlobalMST  = [distGlobalMST ; distNodes];
                end
                
                distGlobal     = [distGlobal ; distNodes];
            end
        end
        ind = ind+1;
    end
    
    %% Root efficiency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i = 1:length(centersTip)
        M = []; Mm = []; k = 1; g = 1;
        for j = 1:length(centersRoot)
            [~,d]     = shortestpath(G,centersTip(i),centersRoot(j));
            if isinf(d) == 0 && d >= 1
                M(k,:) = [d, j];
                k = k+1;
            end
            [~,dm]     = shortestpath(Gmst,centersTip(i),centersRoot(j));
            if isinf(dm) == 0 && dm >= 1
                Mm(g,:) = [dm, j];
                g = g+1;
            end
        end
        
        if isempty(M) == 0
            [D,indj] = min(M(:,1));
            J = M(indj,2);
            sPathRoot = [sPathRoot;D];
            [~,d]     = shortestpath(Gdt,centersTip(i),centersRoot(J));
            sPathRootDT = [sPathRootDT;d];
            distNodes =  round(norm(posNodesTip(i,:)-posNodesRoot(J,:))*conv_px2mum);
            distRoot  = [distRoot ; distNodes];
            
            if isempty(Mm) == 0
                [Dm,~] = min(Mm(:,1));
                sPathRootMST = [sPathRootMST;Dm];
                distRootMST  = [distRootMST ; distNodes];
            end
            
        end
    end
    
    
    save('Analysis\NetworkEfficiency.mat', 'G', 'Gmst', 'Gdt', ...
        'distGlobal' , 'distRoot', 'distGlobalMST' , 'distRootMST', ...
        'sPathGlobal' , 'sPathRoot', 'sPathGlobalMST' , 'sPathRootMST', 'sPathGlobalDT' , 'sPathRootDT', ...
        'Cost', 'CostMST', 'CostDT')
    
else
    Cost=[]; CostMST=[]; CostDT=[];
    save('Analysis\NetworkEfficiency.mat','Cost', 'CostMST', 'CostDT')
end