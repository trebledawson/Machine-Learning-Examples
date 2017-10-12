% ********************************* %
% Naive Bayes Classifier            %
% Glenn Dawson                      %
% 2017-09-25                        %
% ********************************* %

% Input arguments:
% X is an MxN matrix of training data whose rows correspond to instances 
% and whose columns correspond to features
% Y is an Mx1 vector containing known classifications of the corresponding
% rows of X
% Z is an LxN matrix of test data whose rows correspond to instances and 
% whose columns correspond to features

% Output:
% A 1xL vector containing classifications for all instances in Z

function p=naive_bayes(X,Y,Z)

sizeX = size(X);
sizeZ = size(Z);

class_count = unique(Y);

% Mu and sigma are of form ____(class, feature)
mu = zeros(length(class_count),sizeX(2));
sigma = zeros(length(class_count),sizeX(2));

for i=1:length(class_count) % For each class...
    X_class = X(find(Y==i),:);
    size_class = size(X_class);
    
    % Calculate mu and sigma for each feature
    means = mean(X_class);
    stdev = std(X_class);    
    
    for j=1:length(means)
        mu(i,j) = means(j);
        sigma(i,j) = stdev(j);
    end
end

% Create and populate an array consisting of classifications of Z
classifications = zeros(1,length(sizeZ(1)));
for i=1:sizeZ(1) % For each instance...
    posterior = zeros(1,length(class_count));
    for j=1:length(class_count) % For each class...     
        % Compute prior based on frequency of class in data
        prior = numel(X(find(Y==j),:))/numel(X);
        
        % Calculate the class conditional likelihoods
        likelihood = zeros(1,sizeX(2));
        for k=1:sizeX(2) % For each feature...
            likelihood(k) = normpdf(Z(i,k), mu(j,k), sigma(j,k));
        end

        % Compute the posterior distribution assuming class-conditional
        % feature independence
        posterior(j) = prior * prod(likelihood);
    end
    [~,max_index] = max(posterior);
    classifications(i) = max_index;
end
p=classifications;
end