
"""
Created on Mon May 15 22:22:22 2017
@author: longang
"""
function [y_labels]= predict_labels_kNN(w_test, w_train, gnd_Train, k)
    n_test = size(w_test,2); % ex 27x135 -> 135
    n_train = size(w_train,2);
    y_labels = zeros(1,n_test);
    
    dists = compute_distance(w_test, w_train, n_test, n_train);
    
    % For each testing example
    for i=1:n_test
        
        % Sort and select k nearest neighborhoods
       [d, indexes]= sort(dists(i,:),'ascend');
       small_idx = indexes(1:k); 
       small_dist = d(1:k);
       closest_y = gnd_Train(small_idx); % closest k labels
       
       vote_labels = zeros(size(closest_y));
       
       % Let it votes for each label
       for j=1:size(closest_y,2) % for each labels
            label = closest_y(j);
            if ismember(label, find(vote_labels))
                vote_labels(label) = vote_labels(label) + 1;
            else
                vote_labels(label) = 1;
            end
       end
       
       % Break tie if no majority vote, to 1-NN
       nz_labels = vote_labels(vote_labels~=0);
       if all(nz_labels==nz_labels(1)) % if votes are equal for all classes, select min dist class
          small_idx = indexes(1:1);
          most_vote = gnd_Train(small_idx);
       else % Select the most vote label... If votes are not equal for all classes
           [val, vote_lb] = sort(vote_labels,'descend');
           
           big_vote_lb = vote_lb(val==val(1:1)); % 12 14
           [~,loc] = ismember(big_vote_lb,closest_y); % loc of big vote label in closest labels
           [~,d_loc] = min(small_dist(loc)); % min distance loc
           most_vote = big_vote_lb(d_loc);
       end
       y_labels(i)= most_vote;
    end
end

%%
% Euclidean distance between test and training set
 function dists = compute_distance(w_test, w_train, n_test, n_train)
    dists = zeros(n_test,n_train);
    for i=1:n_test
        for j=1:n_train
            dists(i,j) = norm(w_test(:,i) - w_train(:, j));
        end
    end
end