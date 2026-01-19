%Utslipp på alle punkt grunnere enn 200m?
clear all
day2hour = 24;

filename = 'grid_SVIM.nc';
mask_rho = nc_read(filename,'mask_rho');
h = nc_read(filename,'h');

k = 1;
for i = 130:1:200 %Rockall
    for j = 355:1:400
        if h(i,j) < 200
            x1(k) = i;
            y1(k) = j;
            farmid(k) = 11111;
            k = k+1;
        end
    end
end
for i = 270:2:370 %Faroe
    for j = 290:2:370
        if (h(i,j) < 200) && (mask_rho(i,j) == 1)
            x1(k) = i;
            y1(k) = j;
            farmid(k) = 22222;
            k = k+1;
        end
    end
end
for i = 300:2:340 %Iceland
    for j = 465:2:505
        if (h(i,j) < 200) && (mask_rho(i,j) == 1)
            x1(k) = i;
            y1(k) = j;
            farmid(k) = 33333;
            k = k+1;
        end
    end
end
for i = 555:2:634 %Lofoten
    for j = 224:2:250
        if (h(i,j) < 200) && (mask_rho(i,j) == 1)
            x1(k) = i;
            y1(k) = j;
            farmid(k) = 44444;
            k = k+1;
        end
    end
end

fx = x1; 
fy = y1; 

n = length(fx);
mult(1:n) = 5; super(1:n) = 20;
null = zeros(n,1);
en = ones(n,1);
dyp(1:n,1) = 200;

%Egg Buoyancy
mu = 32.48; std = 1.14; %NS Saithe
%mu = 32.41; std = 0.69; %Coastal Cod

for i=1:n
    R(i) = randn(1);
end 
buo = R*std+mu;


fid = fopen('release_svim_egg.rls','w');
for t=1:1:59
    if t < 10
        fprintf(fid,['%d ','1985-0',num2str(2),'-0',num2str(t),' %7.4f %7.4f %2.0f %5.0f %2.0f %7.4f\n'],[mult' fx' fy' dyp farmid' super' buo']');
    elseif (t >= 10) && (t < 29)
        fprintf(fid,['%d ','1985-0',num2str(2),'-',num2str(t),' %7.4f %7.4f %2.0f %5.0f %2.0f %7.4f\n'],[mult' fx' fy' dyp farmid' super' buo']');
    elseif (t >= 29) && (t < 38)
        fprintf(fid,['%d ','1985-0',num2str(3),'-0',num2str(t-28),' %7.4f %7.4f %2.0f %5.0f %2.0f %7.4f\n'],[mult' fx' fy' dyp farmid' super' buo']');
    else
        fprintf(fid,['%d ','1985-0',num2str(3),'-',num2str(t-28),' %7.4f %7.4f %2.0f %5.0f %2.0f %7.4f\n'],[mult' fx' fy' dyp farmid' super' buo']');
    end
end
