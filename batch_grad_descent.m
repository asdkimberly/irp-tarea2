#!/usr/bin/octave-cli --persist

## (C) 2020 Pablo Alvarado
## EL5852 Introducciû˚n al Reconocimiento de Patrones
## Escuela de IngenierûÙa Electrû˚nica
## Tecnolû˚gico de Costa Rica

# --------------------------------------------------------------------
# Linear regression, showing the contours and the estimated line.
#
# WITH BATCH GRADIENT DESCENT, NORMALIZATION AND DATA CENTERING
#
# Implementation of J and gradJ for the general linear case
# --------------------------------------------------------------------

pkg load optim;

%theta0=[0.7 -0.25];

order = 1;



## Data stored each sample in a row, where the last row is the label
D=load("escazu41.dat");

## Construct the design matrix with a 1's column and the original area
Xo=[ones(rows(D),1),D(:,1)];

normalizer_type="normal";

## Normalize the data
nx = normalizer(normalizer_type);
x = nx.fit_transform(Xo);



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


## The outputs vector with the original data
Yo=D(:,4);
ny = normalizer(normalizer_type);
Y = ny.fit_transform(Yo);


## Limits for plot of regressed lines
minArea = min(Xo(:,2));
maxArea = max(Xo(:,2));
minPrice=min(Yo);
maxPrice=max(Yo);

areas=linspace(minArea,maxArea,15); ## Some areas in the whole range
nareas=nx.transform([ones(length(areas),1) areas']); ## Normalized desired areas


## Objective function of the parameters theta
## Requires also the data X (in rows) and corresponding labels Y.
##
## This function is capable of evaluating several sets of theta, each
## one in a row of the given theta.  Hence the result res will have
## as much rows as theta.
function res=J(theta,X,Y)
  ## First compute the residuals for all sets of theta
  R=(X*theta'-Y*ones(1,rows(theta)));
  ## Now square and sum the residuals for each set of theta
  res=0.5*sum(R.*R,1)';
endfunction;

## Gradient of J.
## Analytical solution.
##
## Here we assume that theta has two components only.
## For each theta pair (assumed in a row of the theta matrix) it will
## compute also a row with the gradient: the first column is the partial
## derivative w.r.t theta_0 and the second w.r.t theta_1
function res=gradJ(theta,X,Y)
  res=(X'*(X*theta'-Y*ones(1,rows(theta))))';
endfunction;



%%Debemos saber la cantidad de thetas0 con las que contamos para
%definir el punto inicial del descenso de gradiente

if (order==1) %espacio de error con un espacio de theta de 2D
 th0=-1:0.05:1;   ## Value range for theta0
 th1=-0.5:0.05:2; ## Value range for theta1
 
 [tt0,tt1] = meshgrid(th0,th1);  ## The complete grid
 contwnd = [th0(1) th0(end) th1(1) th1(end)];
 
 theta=[tt0(:) tt1(:)]; ## All theta value pairs in rows
 jj=reshape(J(theta,XX,Y),size(tt0)); ## J values for each pair
 
 ## Precompute the gradient for the chosen grid
 g=gradJ(theta,XX,Y);%% !!OJO!! esta funcion de gradiente así la 
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
 
 ## Learning rate
 alpha = 0.05;   %%OJO ESTO ES UN PARAMETRO EN LA FUNCION DESCENTPOLY
 
 %%EMPIEZA A GENERAR EL DESCENSO DE GRADIENTE, do the learning
 
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


  ## Perform the gradient descent
  ts=t; # sequence of t's

  for i=[1:100] # max 100 iterations
    tc = ts(end,:); # Current position 
    gn = gradJ(tc,XX,Y);  # Gradient at current position
    tn = tc - alpha * gn;# Next position
    ts = [ts;tn];

    if (norm(gn)<0.001) break; endif;
  endfor

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

