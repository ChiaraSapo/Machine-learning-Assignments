%% Assignment 1 - Saporetti Chiara S4798994
clear all
clc 
close all
addpath('Functions')
addpath('Data')

%% Load the weather data set (already converted in numeric form)
% Normal data set
data_set = load("weather_set_numbers_notitle.txt");

% Modified data set to test the smoothing effect
%data_set = load("weather_set_numbers_notitle_mod.txt"); 

%% Loop to compute the mean value of accuracy 
%% To try it: uncomment lines 17-20 and 166-178
%loops=1000;
%testing=zeros(loops,1);
%testing_s=zeros(loops,1);
%for z=1:loops

%% Check that no value of the data set is below 1
for i=1:size(data_set,1)
    for j=1:size(data_set,2)
        if data_set(i,j) < 1 
            disp('Error: at least one value is less then 1');
            return;
        end
    end
end

%% Save number of examples, features, levels for each feature and classes %%%%

% Examples and features
[EXAMPLES, FEATURES] = size(data_set);
FEATURES=FEATURES-1;
LEVELS=zeros(1,FEATURES);

% Levels
for i=1:FEATURES
    temp = unique(data_set(:,i));
    LEVELS(i) = length(temp);        
end

% Classes
CLASSES = length(unique(data_set(:,end)));

% Print the numbers for the user
fprintf('\n\nThis class has %d examples with observations divided in %d different features.\nEach feature can have a different numebr of levels.', EXAMPLES, FEATURES);
fprintf('\nThe result of the observations can be of  %d types.\n', CLASSES)
Feature=(1:1:FEATURES)';
Level=LEVELS';
table_of_Features=table(Feature,Level)

%% Shuffle dataset rows and randomly separate data into subsets %%%%%%%%%%%
%rng('default') % set 0 seed for rand fnct
idx = randperm(EXAMPLES);
data_set = data_set(idx,:);

%% Ask the user how many lines of data he wants to use as training set
prompt = '\nHow many examples do you want to train your classifier?\n';
N_train = input(prompt);

if N_train >14
    disp('Your number exeeds the number of lines');
    return;
elseif N_train==14
    disp('Save some lines to test your classifier!');
    return;
end

% N_train = 10;

%% Divide the data set in a training set and a data set %%%%%%%%%%%%%%%%%%%

% Training set
train_set = data_set(1:N_train, 1:end-1);
train_res = data_set(1:N_train, end);

% Test set
test_set = data_set(N_train+1:end, 1:end-1);
test_res = data_set(N_train+1:end, end);

%% Evaluate the model from the training data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base model
[p_c, p_f_c] = NaiveModel(train_set, train_res, CLASSES, LEVELS);

% Smoothed model
[p_c_smooth, p_f_c_smooth] = NaiveModelSmooth(train_set, train_res, CLASSES, LEVELS);

%% Test the model on test data set %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prob_c = NaiveClassifier(test_set, p_c, p_f_c, CLASSES);
prob_c_smooth = NaiveClassifier(test_set, p_c_smooth, p_f_c_smooth, CLASSES);

% Classifier with log:
%prob_c_log = NaiveClassifierLog(test_set, p_c, p_f_c, CLASSES);
 
%% Show the results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

my_guess=strings(EXAMPLES-N_train,1);
test_res_word=strings(EXAMPLES-N_train,1);

% Base model
for i = 1 : (EXAMPLES-N_train)
        [unused, my_guess_val] = max(prob_c(i, :)); 
        if my_guess_val==2
          my_guess(i)='yes';
        else 
          my_guess(i)='no';
        end
end

my_guess_s=strings(EXAMPLES-N_train,1);

% Smooth model
for i = 1 : (EXAMPLES-N_train)
    
    [unused, my_guess_val] = max(prob_c_smooth(i, :)); 
    if my_guess_val==2
      my_guess_s(i)='yes';
    elseif my_guess_val==1
      my_guess_s(i)='no';
    end
    
    if test_res(i)==2
      test_res_word(i)='yes';
    elseif test_res(i)==1
      test_res_word(i)='no';
    end
    
end

    
%% CASE1: I have the results of the test set
if size(test_set, 2) > size(train_set, 2) - 1 

    accuracy = evaluateAccuracy(prob_c, test_res);
    accuracy_s = evaluateAccuracy(prob_c_smooth, test_res);
    %accuracy_l = evaluateAccuracy(prob_c_log, test_res);
    
    fprintf('My guesses are:');
    table(test_set,test_res_word,my_guess)
    fprintf('Accuracy in guessing is %d%%\n\n\n', accuracy);

    fprintf('My guesses with smoothing are:');
    table(test_set,test_res_word,my_guess_s)
    fprintf('Accuracy in guessing with smoothing is %d%% \n', accuracy_s);

    %fprintf('Accuracy in guessing with log is %d%% \n', accuracy_l);
    
%% CASE2: I don't have the results of the test set: add features
elseif size(test_set, 2) == size(train_set, 2) - 1
    
    fprintf('My guesses are:');
    table(my_guess,test_set)

    fprintf('My guesses woth smoothing are:');
    table(my_guess_s,test_set)

%% CASE3: I have less features than in the data set
else
  disp('Error: in the test set there are less features than in the data set');
  return;
end


% testing(z)=accuracy;
% testing_s(z)=accuracy_s;
% end
% meant=mean(testing)
% meant_s=mean(testing_s)
% plot([1:1:loops],testing) 
% hold on
% plot([1:1:loops],testing_s) 
% hold on 
% yline(meant)
% hold on
% yline(meant_s)
% legend('accuracy','accuracy smooth','mean','mean smooth')
