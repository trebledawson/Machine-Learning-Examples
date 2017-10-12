% ***************************** %
% Logistic Regression Algorithm %
% Glenn Dawson                  %
% 2017-09-28                    %
% ***************************** %
clear all;
tic
% Data and labels generation
cases = 1000;
x1 = mvnrnd([1 1],[0.01 0;0 0.01],cases);
x2 = mvnrnd([1.35 1.35],[0.01 0;0 0.02],cases);

subplot(221)
title('Randomly generated data')
plot(x1(:,1),x1(:,2),'b.');
hold on
plot(x2(:,1),x2(:,2),'r.');
hold off

data = vertcat(x1,x2);
labels = vertcat(zeros(cases,1),ones(cases,1));

% Gradient descent algorithm
cost = 1;
costdiff = 1;
costs = [];
[data_instances, data_features] = size(data);
theta = [0;0;0];
alpha = 0.03;
count = 0;
while abs(costdiff) > 0.0001
    cost = 0;
    sum = [0;0;0];
    for i=1:data_instances
        X = [1;data(i,1);data(i,2)];
        h = (1/(1+exp(-(theta'*X))));
        if i > cases
            sum = sum + ((h - 1)*X);
            cost = cost + (-log(h));
        else
            sum = sum + (h*X);
            cost = cost + (-log(1-h));
        end
    end
    if count > 0
        costdiff = costs(length(costs)) - cost;
    else
        costdiff = abs(cost);
    end
    costs = [costs abs(cost)];
    
    theta = theta - alpha * sum;
    count = count + 1;
end

% Test trained regression on test data
classifications = zeros(1,data_instances);
for i=1:data_instances
    X = [1;data(i,1);data(i,2)];
    h = (1/(1+exp(-(theta'*X))));
    if h < 0.5
        classifications(i) = 0;
    else
        classifications(i) = 1;
    end
end
correct = 0;
incorrect = 0;
for i=1:data_instances
    if classifications(i) == labels(i)
        correct = correct + 1;
    else
        incorrect = incorrect + 1;
    end
end

error = incorrect/data_instances;

subplot(222)
title('scratch classifications of data');
hold on
for i=1:data_instances
    if classifications(i) == 0
        plot(data(i,1),data(i,2),'b.');
    elseif classifications(i) == 1
        plot(data(i,1),data(i,2),'r.');
    end
end

subplot(223)
accuracy = [error,(1-error)];
pielabels = {'Incorrect','Correct'};
pie(accuracy,pielabels);

subplot(224)
for i=1:length(labels)
    labels(i) = labels(i) + 1;
end
B = mnrfit(data,labels);
Bclassifications = zeros(1,data_instances);
for i=1:data_instances
    X = [1;data(i,1);data(i,2)];
    h = (1/(1+exp(-(B'*X))));
    if h < 0.5
        Bclassifications(i) = 1;
    else
        Bclassifications(i) = 2;
    end
end
title('mnrfit classifications of data');
hold on
for i=1:data_instances
    if Bclassifications(i) == 1
        plot(data(i,1),data(i,2),'b.');
    elseif Bclassifications(i) == 2
        plot(data(i,1),data(i,2),'r.');
    end
end
Bcorrect = 0;
Bincorrect = 0;
for i=1:data_instances
    if Bclassifications(i) == labels(i)
        Bcorrect = Bcorrect + 1;
    else
        Bincorrect = Bincorrect + 1;
    end
end

Berror = Bincorrect/data_instances;

% plot(costs,'.');

% ZX = 0.6:0.01:2;
% ZY = 0.6:0.01:2;
% 
% Z = zeros(length(ZX)*length(ZY),2);
% for j=1:length(ZX)
%     for k=1:length(ZY)
%         var = ((j-1)*length(ZY) + k);
%         Z(var,1) = ZX(j);
%         Z(var,2) = ZY(k);
%     end
% end
% 
% Zclass = zeros(1,length(Z));
% for i=1:length(Zclass)
%     X = [1;Z(i,1);Z(i,2)];
%     h = (1/(1+exp(-(theta'*X))));
%     if h < 0.5
%         Zclass(i) = 0;
%     else
%         Zclass(i) = 1;
%     end
% end
% 
% title('Decision boundary map')
% hold on
% for i=1:length(Zclass)
%     if Zclass(i) == 0
%         plot(Z(i,1),Z(i,2),'b.');
%     elseif Zclass(i) == 1
%         plot(Z(i,1),Z(i,2),'r.');
%     end
% end

toc
