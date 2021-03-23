
%pkg load optim;

% Evaluate the hypothesis with all given x
function y=hypothesis(x,theta)
  
%Primero hay que averiguar el orden con el que se trabaja,
%esto se logra con, tener en cuenta que se trabaja en 3D:
  order = length(theta)-1;
  
%Luego generamos 4 hipótesis, una para cada orden (1,2,3,4)
    
  %para orden 1
  if (order=1)
    XX=bsxfun(@power,x,0:length(theta)-1);
    y=XX*theta;
  %para orden 2  
  elseif (order=2)
    XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3)];
    y=XX*theta;
    
  %para orden 3  
  elseif (order=3)
    XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) x.^3 (x(:,1).^2).*x(:,2) (x(:,1).^2).*x(:,3) x(:,1).*(x(:,2).^2) x(:,1).*(x(:,3).^2) (x(:,2).^2).*x(:,3) x(:,2).*(x(:,3).^2) x(:,1).*x(:,2).*x(:,3)];
    y=XX*theta;
  
  %para orden 4  
  elseif (order=4)
    XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) x.^3 (x(:,1).^2).*x(:,2) (x(:,1).^2).*x(:,3) x(:,1).*(x(:,2).^2) x(:,1).*(x(:,3).^2) (x(:,2).^2).*x(:,3) x(:,2).*(x(:,3).^2) x(:,1).*x(:,2).*x(:,3) x.^4 (x(:,1).^3).*x(:,2) (x(:,1).^3).*x(:,3) x(:,1).*(x(:,2).^3) x(:,1).*(x(:,3).^3) (x(:,2).^3).*x(:,3) x(:,2).*(x(:,3).^3) (x(:,1).^2).*(x(:,2).^2) (x(:,1).^2).*(x(:,3).^2) (x(:,2).^2).*(x(:,3).^2) (x(:,1).^2).*x(:,2).*x(:,3) x(:,1).*(x(:,2).^2).*x(:,3) x(:,1).*x(:,2).*(x(:,3).^2)];
    y=XX*theta;
    
  else
    error("El orden debe ser entre 1 y 4, y theta debe tener al menos 2 dimensiones");
  endif
 
endfunction;





