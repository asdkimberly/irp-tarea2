%% Gradient of the loss function
function gtf=gradloss(theta,X,Y)
  gtf=(XX'*(XX*theta'-Y*ones(1,rows(theta))))';
endfunction;
