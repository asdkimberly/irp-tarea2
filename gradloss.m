%% Gradient of the loss function
function gradloss =gtf(theta,X,Y)
  gradloss=(XX'*(XX*theta'-Y*ones(1,rows(theta))))';
endfunction;
