%% Assignment 5 - Saporetti Chiara S4798994

clear all
clc 
close all
addpath('DataANDFunctions')

%% Loading the data
[trainSet1, trainLabels1] = loadMNIST(0, 1); % First digit 
[trainSet2, trainLabels2] = loadMNIST(0, 8); % Second digit 

%% Splitting the data into subsets of different classes
portionOfSubset = [1:150];

Data1 = [trainSet1 trainLabels1];
[n, ~] = size(Data1);
indices1 = randperm(n);
subsetData1 = Data1(indices1(portionOfSubset), :);

Data2 = [trainSet2 trainLabels2];
[m, ~] = size(Data2);
indices2 = randperm(m);
subsetData2 = Data2(indices2(portionOfSubset), :);

% Creating a training set and adapting it to new convention of the data
trainingSets = [subsetData1(:, 1:end-1)', subsetData2(:, 1:end-1)'];
trainingLabels = [subsetData1(:, end)', subsetData2(:, end)'];


%% Training an autoencoder on training set
nh = 2; % Suggested nh because the 2 activations can be used as axes of the 
        % activation space when plotting the data
myAutoencoder = trainAutoencoder(trainingSets, nh);

% Compressed representation of the data
myEncodedData = encode(myAutoencoder, trainingSets);

%% Plotting the data and saving the results
f = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');

plotcl(myEncodedData', trainingLabels');

legend(['Class ', num2str(trainLabels1(1))], ['Class ', num2str(trainLabels2(1))]);
figName=sprintf('Results/Digits_%d_%d.jpg',trainLabels1(1), trainLabels2(1));
saveas(f, figName)