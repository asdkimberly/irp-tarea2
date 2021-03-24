% DESCENTPOLY Execute a gradient descent process
%
% Starting at theta0, search for the minimum of the target function tf
% by iteratively applying a gradient descent process, where the
% gradient of the target function is given by gtf.  The original
% scalar input data is stored in the column vector Xo and the original
% data labels are stored in Yo, i.e. in the house areas/price linear
% regression examples Xo holds the areas and Yo the prices of the
% training data.
%
% The target function tf and its gradient must follow the interface
%
%   function loss = tf(theta,Xo,Yo) , and
%   function grad = gtf(theta,Xo,Yo)
%
% where theta is a vector holding a set of parameters, whose dimension
% specifies the polynomial order.
%
% For instance, for linear regression theta must have two
% dimensions, and if a cubic regression is desired, then theta must have
% four dimensions.  The loss function tf returns a scalar
% and the gradient gtf a vector with  the same number of dimensions
% as the vector theta.
%
% The following parameters are required:
%
% tf: target function computing the loss (or error)
% gtf: gradient of target function
% theta0: initial point for the iteration
% Xo: vector holding the original data (e.g. the house areas)
% Yo: vector holding the original outputs (e.g. the house prices)
% lr: learning rate
%
% The following parameter pairs are optional:
% "method",method: Use "batch","stochastic", "momentum", "rmsprop", "adam"
% "beta",float: beta parameter for momenum (default: 0.7)
% "beta2",float: beta2 parameter for adam (default: 0.99)
% "maxiter",int: maximum number of iterations (default: 200)
% "epsilon",float: tolerance error for convergence (default: 0.001)
% "minibatch",int: size of minibatch (default: 1)
%
% The function should return all intermediate theta values in pos, as
% well as the corresponding error en each of those positions.
%
% Example:
% [pos,errors]=descentpoly(@J,@gradJ,[0 0 0.5],X,Y,0.1,"method","adam","maxIter",10)

%
% (C) 2021 Pablo Alvarado Tarea 2, I Semestre 2021 EL5852
% Introducción al Reconocimiento de Patrones Escuela de Ingeniería
% Electrónica Tecnológico de Costa Rica
%
% (C) 2021 Andr�s Jim�nez Jim�nez y Kimberly  Orozco Retana  
% Tarea 2, I Semestre 2021 
% EL5852 Introducción al Reconocimiento de Patrones 
% Escuela de Ingeniería Electrónica Tecnológico de Costa Rica

% Utilizando c�digo del curso con modificaciones propias
% Sean autores= Andr�s Jim�nez Jim�nez y Kimberly Orozco Retana


pkg load optim;
  
## Define las variables globales importantes

global alpha =0.05; % Learning rate
global beta = 0.7; % Momentum
global beta2 = 0.95; %RMSprop
global rmspepsilon=1e-8; %RMSprop error threshold
global epsilon = 0.005; % Error threshold
global MB=4; % Mini-batch


## Carga los datos
  D=load("escazu40.dat");

## Construir la matriz de pre dise�o, o sea las columans con las 3 entradas: areas, pisos y cuartos; autores
%Xo=[D(:,1),D(:,2),D(:,3)]; %para 3D

## Construye la matriz de prediseno para las areas
  Xo=[D(:,1)]; %Solo areas

## Construye el vector de etiquetas
  Yo=D(:,4);


%[thetas,errors]=descentpoly(@J,@gradJ,[0 0 0.5],Xo,Yo,0.1,"method","adam","maxiter",10)
%function [thetas,errors]=descentpoly(tf,gtf,theta0,Xo,Yo,lr,varargin)

## Define el orden de la regresion polinomial
  order = length(theta0)-1;
  
## Define el tipo de normalizador
  normalizer_type="normal";

## Normaliza los datos de la matriz de prediseño
  nx = normalizer(normalizer_type);  
  x = nx.fit_transform(Xo);

## Se asegura de tener los datos en la forma correcta

  # theta0 must be a row vector
  if (isvector(theta0))
    theta0=theta0(:).';
  else
    error("theta0 must be a row vector");
  endif
    
  # Xo must be a column vector
  if (isvector(Xo))
    Xo=Xo(:);
  else
   error("Xo must be a column vector");
  endif

  ## Yo must be a column vector
  if (isvector(Yo))
    Yo=Yo(:);
  else
    error("Yo must be a column vector");
  endif
  
  
  if (order<1)
    error("El punto inicial de theta0 debe tener al menos 2 dimensiones");
  %%Creamos las matrices de diseño completas; autores
  %para orden 1
  elseif (order==1)
    XX=[x];
    
  %para orden 2  
  elseif (order==2)
    %XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3)]; %Para 3D
    XX=[x x(:,2).^2];
    
  %para orden 3  
  elseif (order==3)
    %XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) x.^3 (x(:,1).^2).*x(:,2) (x(:,1).^2).*x(:,3) x(:,1).*(x(:,2).^2) x(:,1).*(x(:,3).^2) (x(:,2).^2).*x(:,3) x(:,2).*(x(:,3).^2) x(:,1).*x(:,2).*x(:,3)];
    XX=[x x(:,2).^2 x(:,2).^3];
    
  %para orden 4  
  elseif (order==4)
    %XX=[ones(rows(x),1) x x.^2 x(:,1).*x(:,2) x(:,1).*x(:,3) x(:,2).*x(:,3) x.^3 (x(:,1).^2).*x(:,2) (x(:,1).^2).*x(:,3) x(:,1).*(x(:,2).^2) x(:,1).*(x(:,3).^2) (x(:,2).^2).*x(:,3) x(:,2).*(x(:,3).^2) x(:,1).*x(:,2).*x(:,3) x.^4 (x(:,1).^3).*x(:,2) (x(:,1).^3).*x(:,3) x(:,1).*(x(:,2).^3) x(:,1).*(x(:,3).^3) (x(:,2).^3).*x(:,3) x(:,2).*(x(:,3).^3) (x(:,1).^2).*(x(:,2).^2) (x(:,1).^2).*(x(:,3).^2) (x(:,2).^2).*(x(:,3).^2) (x(:,1).^2).*x(:,2).*x(:,3) x(:,1).*(x(:,2).^2).*x(:,3) x(:,1).*x(:,2).*(x(:,3).^2)];
    XX=[x x(:,2).^2 x(:,2).^3 x(:,2).^4];
    
  elseif (order>4)
    error("El límite de orden es de 4");
  endif
  
## Residuo J
function tf=loss(theta,XX,Y)
  ## Calculando el residuo para las tuplas
  R=(hypothesis(XX,theta')-Y*ones(1,rows(theta)));
  ## Sumando el producto punto de los residuos en un escalar
  tf=0.5*sum(R.*R,1)';
endfunction;

## Gradiente analitico
function gtf =gradloss(theta,X,Y)
  gtf=(XX'*(XX*theta'-Y*ones(1,rows(theta))))';
endfunction;
  
%%Debemos saber la cantidad de thetas0 con las que contamos para
%definir el punto inicial del descenso de gradiente

if (order==1) %espacio de error con un espacio de theta de 2D
 th0=-1:0.05:1;   ## Value range for theta0
 th1=-0.5:0.05:2; ## Value range for theta1
 
 [tt0,tt1] = meshgrid(th0,th1);  ## The complete grid
 contwnd = [th0(1) th0(end) th1(1) th1(end)];
 
 theta=[tt0(:) tt1(:)]; ## All theta value pairs in rows
 jj=reshape(loss(theta,XX,Y),size(tt0)); ## J values for each pair
 
 ## Precompute the gradient for the chosen grid
 g=gradloss(theta,XX,Y);%% !!OJO!! esta funcion de gradiente así la 
 %J del método anterior  J(theta,XX,Y) son las funciones gradloss y loss
 %respectivamente
 gjx=reshape(g(:,1),size(tt0));
 gjy=reshape(g(:,2),size(tt1));
 
 ## Show the J surface
 figure(3,"name","J");
 hold off;
 surf(tt0,tt1,jj);
 xlabel('{\theta_0}');
 ylabel('{\theta_1}');
 
 ## Plot the contours in 2D
 figure(1,"name","Contours");
 hold off;
 
 ## Plot the ellipses of the error surface
 contour(tt0,tt1,jj);
 hold on;
 ## and also its the gradient
 quiver(tt0,tt1,gjx,gjy,0.7);
 xlabel("theta_0");
 ylabel("theta_1");
 axis(contwnd);
 daspect([1,1]);

while(1)
  hold on;
 
  printf("Click on countours to set a starting point\n");
  fflush(stdout);

  figure(1,"name","Contours");
  daspect([1,1,1]);

  ## Wait for a mouse click and get the point (t0,t1) in the plot coordinate sys.
  [t0,t1,buttons] = ginput(1);
  t=[t0,t1];%%ESTOS SON LOS ELEMETOS DE THETA0
  gt=gradJ(t,XX,Y);

  ## Clean the previous plot 
  hold off;

  ## Paint first the contour lines
  contour(tt0,tt1,jj);
  hold on;

  ## Add the gradient
  quiver(tt0,tt1,gjx,gjy,0.7);

  xlabel('{\theta_0}');
  ylabel('{\theta_1}');
 
  ## Print some information on the clicked starting point
  printf("J(%g,%g)=%g\n",t0,t1,J(t,XX,Y));
  printf("  GradJ(%g,%g)=[%g,%g]\n",t0,t1,gt(1),gt(2));
  fflush(stdout);

  ## Show the clicked point
  plot([t0],[t1],"*r");

  axis(contwnd);
  daspect([1,1]);

if (method=="batch")
  ## Perform the gradient descent
    ts=t; # sequence of t's

    for i=[1:100] # max 100 iterations
    tc = ts(end,:); # Current position 
    gn = gradloss(tc,XX,Y);  # Gradient at current position
    tn = tc - alpha * gn;# Next position
    ts = [ts;tn];

      if (norm(gn)<0.001) break; endif;
    endfor;

  # Draw the trajectory
  plot(ts(:,1),ts(:,2),"k-");
  plot(ts(:,1),ts(:,2),"ob");

  # Paint on a second figure the corresponding line
  figure(2,"name","Regressed line");
  hold off;
  plot(Xo(:,2),Yo,"*b");
  hold on;
  
  ## We have to de-normalize the normalized estimation
  nprices = nareas * ts(1,:)';%LO HACE solo con el elemento de la primera fila de ts
  prices=ny.itransform(nprices);
 
  plot(areas,prices,'k',"linewidth",2);

  ## and now with the intermediate versions
    for (i=[2:rows(ts)])%lo hace desde el elemeto de la segunda fila de ts hasta el final de ts
    nprices = nareas * ts(i,:)';
    prices=ny.itransform(nprices); 	
    plot(areas,prices,'r',"linewidth",1);
    endfor;
  ## Repaint the last one as green
  plot(areas,prices,'g',"linewidth",3);

  axis([minArea maxArea minPrice maxPrice]);  
endwhile;


elseif (method=="stoch")
 ## Perform the gradient descent
  ts=t; # sequence of t's

  j=0;
  for i=[1:500] # max 500 iterations
    tc = ts(end,:); # Current position
    sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples
    gn = gradJ(tc,X(sample,:),Y(sample));  # Gradient at current position
    tn = tc - alpha * gn;# Next position
    ts = [ts;tn];

    if (norm(tc-tn)<epsilon)
      j=j+1;
      if (j>5) ## Only exit if several times the positions have been close enough
        break;
      endif;
    else
      j=0;
    endif;
  endfor

  # Draw the trajectory
  plot(ts(:,1),ts(:,2),"k-");
  plot(ts(:,1),ts(:,2),"ob");

  printf("  Number of steps (no momentum): %i\n",rows(ts));
  
  # Paint on a second figure the corresponding line
  figure(2,"name","Regressed line");
  hold off;
  plot(Xo(:,2),Yo,"*b");
  hold on;
  
  ## We have to de-normalize the normalized estimation
  nprices = nareas * ts(1,:)';
  prices=ny.itransform(nprices);

  plot(areas,prices,"--;initial line;","linewidth",2,"color",[0.5,0.5,0.5]);

  ## and now the last line
  nprices = nareas * ts(end,:)';
  prices=ny.itransform(nprices); 	
  plot(areas,prices,'g;vanilla;',"linewidth",3);

  axis([minArea maxArea minPrice maxPrice]);
endwhile;

elseif (method=="momentum")
 ## Perform the gradient descent
  ts=t; # sequence of t's

  sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples for init.
  V = gradJ(t,X(sample,:),Y(sample));  # Gradient at current position
    
  
  j=0;
  for i=[1:500] # max 500 iterations
    tc = ts(end,:); # Current position
    sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples
    gn = gradJ(tc,X(sample,:),Y(sample));  # Gradient at current position

    V = beta*V + (1-beta)*gn; ## Filter the gradient
    tn = tc - alpha * V;      ## Gradient descent with filtered grad
    ts = [ts;tn];

    if (norm(tc-tn)<epsilon)
      j=j+1;
      if (j>5) ## Only exit if several times the positions have been close enough
        break;
      endif;
    else
      j=0;
    endif;
  endfor

  ## Draw the trajectory
  figure(1);
  plot(ts(:,1),ts(:,2),"k-");
  plot(ts(:,1),ts(:,2),"og");

  printf("  Number of steps (with momentum): %i\n",rows(ts));
  
  # Paint on a second figure the corresponding line
  figure(2,"name","Regressed line");
  hold on;
  
  ## Final result line
  nprices = nareas * ts(end,:)';
  prices=ny.itransform(nprices); 	
  plot(areas,prices,'b;with momentum;',"linewidth",3);

  axis([minArea maxArea minPrice maxPrice]);
endwhile;

elseif (method=="rmsprop")
## Perform the gradient descent
  ts=t; # sequence of t's

  sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples for init.
  gn = gradJ(t,X(sample,:),Y(sample));  # Gradient at current position
  s = gn.^2;
    
  
  j=0;
  for i=[1:500] # max 500 iterations
    tc = ts(end,:); # Current position
    sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples
    gn = gradJ(tc,X(sample,:),Y(sample));  # Gradient at current position
    s = beta2*s + (1-beta2)*(gn.^2);
    gg = gn./(sqrt(s + rmspepsilon) );

    tn = tc - alpha * gg;      ## Gradient descent with filtered grad
    ts = [ts;tn];

    if (norm(tc-tn)<epsilon)
      j=j+1;
      if (j>2) ## Only exit if several times the positions have been close enough
        break;
      endif;
    else
      j=0;
    endif;
  endfor

  ## Draw the trajectory
  figure(1);
  plot(ts(:,1),ts(:,2),"k-");
  plot(ts(:,1),ts(:,2),"og");

  printf("  Number of steps (with RMSprop): %i\n",rows(ts));
  
  # Paint on a second figure the corresponding line
  figure(2,"name","Regressed line");
  hold on;
  
  ## Final result line
  nprices = nareas * ts(end,:)';
  prices=ny.itransform(nprices); 	
  plot(areas,prices,'b;with RMSprop;',"linewidth",3);

  axis([minArea maxArea minPrice maxPrice]);
endwhile;

elseif (method=="adam")
 ## Perform the gradient descent
  ts=t; # sequence of t's

  sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples for init.
  gn = gradJ(t,X(sample,:),Y(sample));  # Gradient at current position
  s = gn.^2; # Initialize to avoid bias
  V = gn;
    
  
  j=0;
  for i=[1:500] # max 500 iterations
    tc = ts(end,:); # Current position
    sample=round(unifrnd(1,rows(X),MB,1)); # Use MB random samples
    gn = gradJ(tc,X(sample,:),Y(sample));  # Gradient at current position
    s = beta2*s + (1-beta2)*(gn.^2);
    V = beta*V + (1-beta)*gn;
    gg = V./(sqrt(s + rmspepsilon) );

    tn = tc - alpha * gg;      ## Gradient descent with filtered grad
    ts = [ts;tn];

    if (norm(tc-tn)<epsilon)
      j=j+1;
      if (j>5) ## Only exit if several times the positions have been close enough
        break;
      endif;
    else
      j=0;
    endif;
  endfor

  ## Draw the trajectory
  figure(1);
  plot(ts(:,1),ts(:,2),"k-");
  plot(ts(:,1),ts(:,2),"og");

  printf("  Number of steps (with ADAM): %i\n",rows(ts));
  
  # Paint on a second figure the corresponding line
  figure(2,"name","Regressed line");
  hold on;
  
  ## Final result line
  nprices = nareas * ts(end,:)';
  prices=ny.itransform(nprices); 	
  plot(areas,prices,'b;with ADAM;',"linewidth",3);

  axis([minArea maxArea minPrice maxPrice]);
endwhile;
endif;
 
 
##############################################################################################
## A partir de acá el codigo no tiene relacion 
 
  tf = loss(theta,Xo,Yo);
  gtf = gradloss(XX,theta0);
  
  defaultMethod="batch";
  defaultBeta=0.7;
  defaultBeta2=0.99;
  defaultMaxIter=200;
  defaultEps=0.001;
  defaultMinibatch=1;

  p = inputParser;
  validMethods={"batch","stochastic","momentum","rmsprop","adam"};
  checkMethod = @(x) any(validatestring(x,validMethods));
  addParameter(p,'method',defaultMethod,checkMethod);

  checkBeta = @(x) isreal(x) && isscalar(x) && x>=0 && x<=1;
  checkRealPosScalar = @(x) isreal(x) && isscalar(x) && x>0;

  addParameter(p,'beta',defaultBeta,checkBeta);
  addParameter(p,'beta2',defaultBeta2,checkBeta);
  addParameter(p,'maxiter',defaultMaxIter,checkRealPosScalar);
  addParameter(p,'epsilon',defaultEps,checkRealPosScalar);
  addParameter(p,'minibatch',defaultMinibatch,checkRealPosScalar);
  
  parse(p,varargin{:});
  
  if ~checkBeta(lr)
    error("Learning rate must be between 0 and 1");
  endif

  ## ################################################################
  ## Your code in here!!!
  ##
  ## Next lines are just an example.  You should change them
 
 
elseif (order==2) %espacio de error con un espacio de theta de 3D
 th0=theta0(1,1);
 th1=theta0(1,2);
 th2=theta0(1,3);

elseif (order==3) %espacio de error con un espacio de theta de 4D
 th0=theta0(1,1);
 th1=theta0(1,2);
 th2=theta0(1,3);
 th3=theta0(1,4);

elseif (order==4) %espacio de error con un espacio de theta de 5D
 th0=theta0(1,1);
 th1=theta0(1,2);
 th2=theta0(1,3);
 th3=theta0(1,4);
 th4=theta0(1,5);
endif
  endfunction;

  method = p.Results.method;       ## String with desired method
  beta = p.Results.beta;           ## Momentum parameters beta
  beta2 = p.Results.beta2;         ## ADAM paramter beta2
  maxiter = p.Results.maxiter;     ## maxinum number of iterations
  epsilon = p.Results.epsilon;     ## convergence error tolerance
  minibatch = p.Results.minibatch; ## minibatch size
  
  
  
  
  
  thetas = [theta0];
  errors = [tf(theta0,Xo,Yo)];
  


