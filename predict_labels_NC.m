"""
Created on Mon May 15 22:22:22 2017
@author: longang
"""
function [y_labels]= predict_labels_NC(w_test, w_train, gnd_Train)
    n_test = size(w_test,2);
    y_labels = zeros(1,n_test);
    
    % Find class centroid of 15 person
    unique_faceIdx = unique(gnd_Train); % returns 1 2 3 4 5
    num_person = size(unique_faceIdx,2); % returns 15 
    class_centroid = zeros(size(w_train,1), num_person); % ex: 27x15
    
    for i=unique_faceIdx % 1 2 3 ... 15
        person_idx = gnd_Train==i; % returns 1 2 3 for 2 2 2, if i=2
        w_train_cluster = w_train(:, person_idx); % returns w_train of each person
        % class centroid for person i
        class_centroid(:,i) = mean(w_train_cluster,2); % 27x1, return class centroid 
    end
    
    % Euclidean distance from testing examples to class centroids
    dists = compute_distance(w_test, class_centroid, n_test, unique_faceIdx);
    
    % Select minimum distance for each testing example
    for i=1:n_test
       [~, p_idx]= min(dists(i,:)); % 1 testing example to every class centroids
       y_labels(i) = p_idx; % p_idx is a label of gnd_Train
    end
end

%%
% Euclidean distance between test and training set
 function dists = compute_distance(w_test, class_centroid, n_test, unique_faceIdx)
    dists = zeros(n_test,15);
    for i=1:n_test
        for j=unique_faceIdx % j is label of person
            dists(i,j) = norm(w_test(:,i) - class_centroid(:,j));
        end
    end
end