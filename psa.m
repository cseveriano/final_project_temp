function [ IDX , MEMBERS, MAX, DIM] = psa( X, k )
%PSA Part and Select Algorithm for clustering 
%
% IDX = PSA(X,K) partitions the points in the N-by-P data matrix X
%    into K clusters.  This partition minimizes the diameter, over all clusters, of
%    the maximum difference between points in a certain dimension P of X.
%    Rows of X correspond to points, columns correspond to variables.  
%    Note: when X is a vector, PSA treats it as an N-by-1 data matrix, 
%    regardless of its orientation.  PSA returns an N-by-1 vector IDX 
%    containing the cluster indices of each point.  By default, PSA uses 
%    squared Euclidean distances.
%
% [IDX, MEMBERS] = PSA(X,K) returns the K more representative MEMBERS 
%
%
% [IDX, MEMBERS, MAX] = PSA(X,K) returns the K cluster maximum diameter difference
%    MAX for each cluster.
%
% [IDX, MEMBERS, MAX, DIM] = PSA(X,K) returns the dimension of the K cluster 
%    maximum diameter difference
%
% Examples: 
% 
%     X = [randn(30,2)+2.5;randn(30,2)-2.5];
%     subplot(1,2,1); plot(X(:,1),X(:,2),'bo');
%     IDX = psa(X,4);
%     subplot(1,2,2); plot(X(IDX==1,1),X(IDX==1,2),'bo');
%     hold on;
%     plot(X(IDX==2,1),X(IDX==2,2),'ro');
%     plot(X(IDX==3,1),X(IDX==3,2),'go');
%     plot(X(IDX==4,1),X(IDX==4,2),'ko');
%
%     F = (exp(1-(rand(200,1)*(7)+1)/4*(linspace(-1,1,55)))+((rand(200,1)*(7)+1)*ones(1,55)).*sin((rand(200,1)*(7)+1)*(linspace(-1,1,55))));
%     subplot(1,2,1);
%     plot(linspace(-1,1,55),F); xlim([-1 1]);
%     IDX = psa(F,6);
%     colors = {'b-' 'g-' 'r-' 'c-' 'm-' 'y-'};
%     subplot(1,2,2);
%     for i=1:6 plot(linspace(-1,1,55),F(IDX==i,:),colors{i}); hold on; xlim([-1 1]); end
%     
% 
% See also: kmeans
%
% References:
% Shaul Salomon, Gideon Avigad, Alex Goldvard, and Oliver Schuetze, 
% PSA ? A New Scalable Space Partition Based Selection Algorithm for MOEAs.
% In the proceedings of EVOLVE 2012.
%
% $Author: Alan de Freitas $    $Date: 2012/06/11 $    $Revision: 1.0 $
% Copyright: 2012
% http://www.mathworks.com/matlabcentral/fileexchange/authors/255737

% If X has only one row
if (size(X,1)==1)
    X = X';
end

% Initially, all points are in cluster 1
IDX = ones(size(X,1),1);
MAX = zeros(1,k);
DIM = zeros(1,k);
[MAX(1) DIM(1)] = max(max(X(IDX==1,:))-min(X(IDX==1,:)),[],2);
lastcluster = [];

% For each additional cluster
for i=2:k
    % Find max diameters of the last clusters split
    if sum(IDX==(i-1))==0
        break;
    else
        for j=lastcluster
            [MAX(j) DIM(j)] = max(max(X(IDX==j,:))-min(X(IDX==j,:)),[],2);
        end
    end
    
    % Group with largest max diameter so far is divided
    [~,to_divide] = max(MAX(1:i-1));
    
    % The ones which are over 50% in the dividing dimension go the new
    % cluster
    IDX(((X(:,DIM(to_divide))-min(X(IDX==to_divide,DIM(to_divide))))>=(MAX(to_divide)/2))&(IDX==to_divide)) = i;
    
    % Update which the last clusters
    lastcluster = [i, to_divide];
end

% Finds the representative group
if (nargout > 1)
    for j=lastcluster
        [MAX(j) DIM(j)] = max(max(X(IDX==j,:))-min(X(IDX==j,:)),[],2);
    end
    MEMBERS = zeros(1,max(IDX));
    for i=1:max(IDX)
        % cluster i has diameter MAX(i) in the DIM(i) dimension
        A = (((X(:,DIM(i))-min(X(IDX==i,DIM(i))))-(MAX(i)/2)));
        A(IDX~=i) = inf;
        [~,MEMBERS(i)] = min(abs(A));
    end
end  

end

