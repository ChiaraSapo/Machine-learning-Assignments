%% Assignment 2 - Saporetti Chiara S4798994

clear all
clc 
close all
addpath('Functions')
addpath('Data')

% Number of models we use
N_MODELS=3;

%% Task 1: load datasets
CarsDataset=readtable('mtcarsdata-4features.csv');
TurkishDataset=load('turkish-se-SP500vsMSCI.csv');

%% Task 2
%% 1 ) One-dimensional problem without intercept on the Turkish dataset
% x:data
% t:result
x_Turk=TurkishDataset(:,1);
t_Turk=TurkishDataset(:,2);
% Call 1D linear regression function
w1=oneDimLinReg(x_Turk,t_Turk);
y1=x_Turk*w1;

% Plot
figure; 
plot(x_Turk,t_Turk,'bx')
hold on;
plot(x_Turk,y1,'red')
title('1D problem without intercept on the Turkish dataset')
xlabel('x (data)'); 
ylabel('t (target)'); 

%% 2 ) Compare graphically the solution obtained on different random 
%% subsets (10%) of the whole Turkish data set
% Plot model
figure; 
hold on;
plot(x_Turk,t_Turk,'bx')
plot(x_Turk,y1,'red')
title('1D problem of different subsets (10% of dataset)')
xlabel('x (data)'); 
ylabel('t (target)'); 

for i=1:7
    % Pick random lines of the dataset
    idx = randperm(length(x_Turk));
    subSet = TurkishDataset(idx(1:round(length(x_Turk)/10)),:);
    x2=subSet(:,1);
    t2=subSet(:,2);
    % Call 1D linear regression function
    w=oneDimLinReg(x2,t2);
    y2=w*x2;
    
    % Plot
    hold on;
    plot(x2,y2,'green')
    legend('Data','Model of dataset', 'Model of subsets');
end


%% 3 ) One-dimensional problem with intercept on the Motor Trends car data, 
%% using columns mpg and weight
% x: weight
% t: mpg
x_Car=CarsDataset{:,5};
t_Car=CarsDataset{:,2};
% Call 1D linear regression function with intercept
[w1,w0]=oneDimLinReg_intercept(x_Car,t_Car);
y3=w1*x_Car+w0;

% Plot
figure; 
plot(x_Car,t_Car,'bx')
hold on;
plot(x_Car,y3,'red')
title('1D problem with intercept on the car data, using mpg and weight')
xlabel('weight (data)'); 
ylabel('mpg (target)'); 

%% 4 ) Multi-dimensional problem on the complete MTcars data, using all 
%% four columns (predict mpg with the other three columns)
% x: all variables
% t: mpg
X_Car=CarsDataset{:,3:5};
t_Car=CarsDataset{:,2};
% Call multi-D linear regression function
W=multiDimLinReg(X_Car,t_Car);
y4=X_Car*W;

MultidimResults = table(t_Car, y4);
MultidimResults.Properties.VariableNames = {'Real Target t' 'Predicted Target y'};

figure
uitable('Data',MultidimResults{:,:},'ColumnName', MultidimResults.Properties.VariableNames,...
    'Units', 'Normalized', 'Position',[0, 0, 1, 1]);



%% Task 3: re-run 1,3,4 on a training set (5% of dataset) and test it on a 
%% test set (95% of dataset). Compute the square errors.

LOOPS=10;
obj.training=zeros(LOOPS,N_MODELS);
obj.test=zeros(LOOPS,N_MODELS);

for i=1:LOOPS
    % Divide the sets 
    idx = randperm(size(CarsDataset,1));
    VAR=round(size(CarsDataset,1)/20);
    CarTrainingSet = CarsDataset{idx(1:VAR),2:end}; % first part: 5%
    CarTestSet = CarsDataset{idx(VAR:end),2:end}; % second part: 95%

    idx2 = randperm(size(TurkishDataset,1));
    VAR2=round(size(TurkishDataset,1)/20);
    TurkTrainingSet = TurkishDataset(idx(1:VAR2),:);
    TurkTestSet = TurkishDataset(idx(VAR2:end),:);
    
    % Print dimensions of training sets
    if i==1
        fprintf('\nTraining set for cars dataset is of size %d', VAR)
        fprintf('\nTraining set for turkish dataset is of size %d', VAR2)
    end
    
    %% Compute model and J from training sets
    %% 1
    x_1=TurkTrainingSet(:,1);
    t_1=TurkTrainingSet(:,2);
    w1_1=oneDimLinReg(x_1,t_1);
    obj.training(i,1) = meanSquareError(x_1,t_1,w1_1,0,1);

    %% 3
    x_2=CarTrainingSet(:,4);
    t_2=CarTrainingSet(:,1);
    [w1_2,w0]=oneDimLinReg_intercept(x_2,t_2);
    obj.training(i,2) = meanSquareError(x_2,t_2,w1_2,w0,1);
    
    %% 4
    x_3=CarTrainingSet(:,2:4);
    t_3=CarTrainingSet(:,1);
    w1_3=multiDimLinReg(x_3,t_3);
    obj.training(i,3) = meanSquareError(x_3,t_3,w1_3,0,2);
    
    %% Test results and compute J from training sets
    %% 1
    x=TurkTestSet(:,1);
    t=TurkTestSet(:,2);
    obj.test(i,1) = meanSquareError(x,t,w1_1,0,1);
     
    %% 3
    x=CarTestSet(:,4);
    t=CarTestSet(:,1);
    obj.test(i,2) = meanSquareError(x,t,w1_2,w0,1);
    
    %% 4
    x=CarTestSet(:,2:4);
    t=CarTestSet(:,1);
    obj.test(i,3) = meanSquareError(x,t,w1_3,0,2);
    
end 

% Compute average of the objectives
average_obj.training=zeros(N_MODELS,1);
average_obj.test=zeros(N_MODELS,1);

for i=1:N_MODELS
    average_obj.training(i)=mean(obj.training(i));
    average_obj.test(i)=mean(obj.test(i));
end

ObjectivesResults = table(average_obj.training, average_obj.test);
ObjectivesResults.Properties.VariableNames = {'Training' 'Test'};
ObjectivesResults.Properties.RowNames = {'1-D' '1-D offset' 'multi-D'};

figure
uitable('Data',ObjectivesResults{:,:},'ColumnName',ObjectivesResults.Properties.VariableNames,...
    'RowName',ObjectivesResults.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

