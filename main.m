
"""
Created on Mon May 15 22:22:22 2017
@author: longang
"""
clear all;
clc;

%%
% load main dataset
ds_path = 'Yale_32x32.mat';
load(ds_path); % fea=165x1024, gnd=165x1

% normalize pixel values to [0,1]
fea = fea/255;

fea = transpose(fea); % Transpose to column vector 1024x165
gnd = transpose(gnd); % Transpose to 1x165
  
%%
% There are 7 subsets : p=(2,3,4,5,6,7,8)
% Each subset contains 50 splits
subset_path = '2Train/';
subset_num = 2; % p = 2
k_big_eig = 27; % 27 for p=2; 43 for p=3; 59 for p=4; 70 for p=5; 80 for p=6; 90 for p=7; 95 for p=8

classifier = 1; % 1 for NC, 2 for k-NN
k = 2; % default k =2 for k-NN

num_splits = 50; % mat files in each subset

is_display = 0;
error_each_split = zeros(num_splits,1); % 50 splits in each subse

%%

for n=1:num_splits
    
    % Step 1: Load each split
    load(sprintf(strcat(subset_path,'%d.mat'), n)); % trainIdx=30x1, testIdx=135x1
    
    fea_Train = fea(:, trainIdx); % 1024x30
    gnd_Train = gnd(trainIdx); % 1x30
    fea_Test = fea(:,testIdx); % 1024x135
    gnd_Test = gnd(testIdx); % 1x135
    
    num_train = size(fea_Train,2);
    num_test = size(fea_Test,2);
    
    if(is_display == 1)
         show_some(fea_Train,10,2,'Training examples');
    end
    
%%    
    % Step 2: Calculate mean of training data (mean)
    mean_face = mean(fea_Train,2);
    if(is_display == 1)
         show_img(mean_face);
    end
    
%%  
    % Step 3: Subtract mean from each face
    noMean_train = fea_Train - repmat(mean_face,[1,num_train]);
    if(is_display == 1)
        show_some(noMean_train,10,2, 'Centered faces');
    end
    
%%
    % Step 4: Compute the covariance matrix C=A'A
    C = transpose(noMean_train) * noMean_train; % generates 30x30 matrix
    
%%    
    % Step 5: Compute eigenvectors and eigenvalues of C
    [eigenVecC, eigenValC] = eig(C); % generate 30x30 eigenvectors
    
%%
    % Step 6: Compute eigenvectors of L=AA' from eigenvectors of C (eigenVecL = A*eigenVecC_K)
    %       (due to number of samples are in rows, AA' will be smaller than A'A)
    eigenVecL = noMean_train*eigenVecC; % is the eigenface
    
            % Note: The M eigenvalues of C=A'A (along with their corresponding eigenvectors) 
            % correspond to the M largest eigenvalues of L=AA' (along with their corresponding eigenvectors)
            
%%  
  % Step 7: Select K largest eigenvalues of L that correspond to C (eigenValL=eigenValC)
    all_eigenValC = diag(eigenValC); % generate 30 eigenvalues
    [val,idx_descendingOrder] = sort(all_eigenValC,'descend'); 
    eigenVecL = eigenVecL(:,idx_descendingOrder);
    eigenVecL_K = eigenVecL(:,1:k_big_eig);
    
    if(is_display == 1)
        show_some(eigenVecL_K,10,2, 'Eigenfaces');
    end
        % Step 7.1: Normalize eigenfaces
        for j=1:k_big_eig
            eigenVecL_K(:,j) = eigenVecL_K(:,j)/norm(eigenVecL_K(:,j));
        end
    
    if(is_display == 1)
        show_some(eigenVecL_K,10,2, 'Eigenfaces (normalized ||u||==1)');
    end
    
%%  
    % Step 8: Compute weight(or coefficience) of every training examples related to K
    % largest eigenvectors

    w_train = transpose(eigenVecL_K) * noMean_train;
    % End Training
    
%%
    % Test Data
    
    % Step 1: Substract mean from all testing images
    noMean_test = fea_Test - repmat(mean_face,[1,num_test]);
    
    % Step 2: Compute the weights(or coefficience) of test examples
    w_test = transpose(eigenVecL_K) * noMean_test;
    
%%
    % Step 3: Classify testing images using 1-NN and k-NN
    if(classifier ==1) % NC classifier
        p_labels = predict_labels_NC(w_test, w_train, gnd_Train);
    else % k-NN classifier

        %load fisheriris
        %Mdl = fitcknn(w_train',gnd_Train','NumNeighbors',4);
        %p_labels = predict(Mdl,w_test');
        
        p_labels = predict_labels_kNN(w_test, w_train, gnd_Train, k);
    end
    
    incorrect_classify = sum(p_labels ~= gnd_Test);
    incorrect_rate = (incorrect_classify/num_test)*100;
    error_each_split(n) = incorrect_rate;
    
    fprintf('Incorrect classify (%d/%d)\n', incorrect_classify, num_test);
    fprintf('Error rate for split(%d):%0.2f\n',n, incorrect_rate);
    fprintf('----------------------------------- \n');
    
end

%%
% Margin difference

mu = sum(error_each_split)/num_splits; % mean error of 50 splits
varr = sum((error_each_split-mu).^2)/num_splits;
std_dev = sqrt(varr);

fprintf('\nAverage error rate for subset (%d): %0.2f (+/- %0.2f)\n', subset_num, mu, std_dev);

