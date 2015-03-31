function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Closest centroid corresponding to each X member
idx = zeros(size(X,1), 1);

min = -1;
% for each index (or X member)
for i = 1:size(idx, 1) 
    min = norm(X(i, :) - centroids(1,:), 2); % default min distance = from this x to centroid 1
    idx(i) = 1; % default centroid of this index
    % for each centroid    
    for j = 2:K 
	% calculate distance from each X member to this centroid
        distance = norm(X(i,:) - centroids(j,:), 2);
	if distance < min
	    min = distance; 
	    idx(i) = j; % this k is centroid
	endif
    endfor
endfor

% =============================================================

end

