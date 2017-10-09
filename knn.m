% ********************* %
% K-Nearest Neighbor    %
% Glenn Dawson          %
% 2017-09-27            %
% ********************* %

function p=knn(X,Y,Z,K)

% Input arguments:
% X is an MxN matrix of training data whose rows correspond to instances 
% and whose columns correspond to features
% Y is an Mx1 vector containing known classifications of the corresponding
% rows of X
% Z is an LxN matrix of test data whose rows correspond to instances and 
% whose columns correspond to features
% K is the KNN parameter

% Output:
% A 1xL vector containing classifications for all instances in Z

[X_instances, ~]=size(X);
[Z_instances, ~]=size(Z);

classifications = zeros(1,Z_instances);
for i=1:Z_instances % For all test instances...
    
    % Find distances from test instance to training instance
    distances = zeros(1,X_instances);
    for j=1:X_instances % For all training instances...
        distances(j) = pdist([X(j,:);Z(i,:)],'euclidean');
    end
    
    % Find the K nearest neighbors
    K_nearest = zeros(1,K);
    for j=1:K
        [~,min_index] = min(distances);
        K_nearest(j) = min_index;
        distances(min_index) = 0;
    end
    
    % Find the nearest neighbor
    nearest = mode(K_nearest);
    
    % Select the class corresponding to the nearest neighbor
    classifications(i) = Y(nearest);
end

p=classifications;
end
