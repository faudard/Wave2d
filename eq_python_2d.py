"""
    eq_python_2d.py 
    
    2D WAVE-EQUATION using  explicite finite difference method

    2D WAVE-EQUATION: u_{tt} = c *c* ( u_{xx} + u_{yy} )

                                F. Audard
"""



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


class wave2d(object):
    def __init__(self,height,width,T,nx,ny,nt,c):
        
        self.x = np.linspace(-0.5*width,0.5*width,nx)
        self.y = np.linspace(-0.5*height,0.5*height,ny)
        self.t = np.linspace(0,T,nt+1)

        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.dt = self.t[1]-self.t[0]
        
        self.xx,self.yy = np.meshgrid(self.x,self.y)

        # Gamma_x squared
        self.gx2 = c*self.dt/self.dx

        # Gamma_y squared
        self.gy2 = c*self.dt/self.dy

        # 2*(1-gamma_x^2-gamma_y^2)
        self.gamma = 2*(1 - self.gx2 - self.gy2)
        

    def solve(self,ffun,gfun):
        BC =1
        f = ffun(self.xx,self.yy)
        g = gfun(self.xx,self.yy) 

        u = np.zeros((nx,ny,nt+1))

        # Set initial condition
        u[:,:,0] = f

        """ Compute first time step """
        u[:,:,1] = 0.5*self.gamma*f+self.dt*g
        u[1:-1,1:-1,1] += 0.5*self.gx2*(f[1:-1,2:]+f[1:-1,:-2])
        u[1:-1,1:-1,1] += 0.5*self.gy2*(f[:-2,1:-1]+f[2:,1:-1])

        for k in range(1,nt):
            # Every point contains these terms
            u[:,:,k+1] = self.gamma*u[:,:,k] - u[:,:,k-1]

            # Interior points
            u[1:-1,1:-1,k+1] += self.gx2*(u[1:-1,2:,k]+u[1:-1,:-2,k]) + \
                                self.gy2*(u[2:,1:-1,k]+u[:-2,1:-1,k])            

            # Dirchlet condition
            #    if bc['W'] is None:   
            if BC == 1:
            # Top boundary
                        u[1,1:-1,k+1] += 0
                                      
                        # Right boundary
                        u[1:-1,-2,k+1] += 0

                        # Bottom boundary
                        u[-1,1:-2,k+1] +=  0 
                               
                        # Left boundary
                        u[1:-1,1,k+1] += 0

                        # Top right corner
                        u[1,-2,k+1] += 0

                        # Bottom right corner
                        u[-2,-2,k+1] += 0

                        # Bottom left corner
                        u[-2,1,k+1] += 0
    
                        # Top left corner
                        u[1,1,k+1] += 0  
            # Neumann condition
            elif BC ==2:

                        # Top boundary
                        u[0,1:-1,k+1] +=  2*self.gy2*u[1,1:-1,k] + \
                               self.gx2*(u[0,2:,k]+u[0,:-2,k])
                                      
                        # Right boundary
                        u[1:-1,-1,k+1] += 2*self.gx2*u[1:-1,-2,k] + \
                               self.gy2*(u[2:,-1,k]+u[:-2,-1,k])

                        # Bottom boundary
                        u[-1,1:-1,k+1] +=  2*self.gy2*u[-2,1:-1,k] + \
                                self.gx2*(u[-1,2:,k]+u[-1,:-2,k]) 
                               
                        # Left boundary
                        u[1:-1,0,k+1] += 2*self.gx2*u[1:-1,1,k] + \
                              self.gy2*(u[2:,0,k]+u[:-2,0,k])

                        # Top right corner
                        u[0,-1,k+1] += 2*self.gx2*u[0,-2,k] + \
                          2*self.gy2*u[1,-1,k]

                        # Bottom right corner
                        u[-1,-1,k+1] += 2*self.gx2*u[-1,-2,k] + \
                           2*self.gy2*u[-2,-1,k]

                        # Bottom left corner
                        u[-1,0,k+1] += 2*self.gx2*u[-1,1,k] + \
                          2*self.gy2*u[-2,0,k]
    
                        # Top left corner
                        u[0,0,k+1] += 2*self.gx2*u[0,1,k] + \
                         2*self.gy2*u[1,0,k]    
            elif BC ==3:
                        # Top boundary
                        u[0,1:-1,k+1] +=  u[1,1:-1,k] 
                                    
                        # Right boundary
                        u[1:-1,-1,k+1] += u[1:-1,-2,k] 

                        # Bottom boundary
                        u[-1,1:-1,k+1] +=  u[-2,1:-1,k]
                               
                        # Left boundary
                        u[1:-1,0,k+1] += u[1:-1,1,k] 

                        # Top right corner
                        u[0,-1,k+1] += u[0,-2,k]

                        # Bottom right corner
                        u[-1,-1,k+1] += u[-1,-2,k] 

                        # Bottom left corner
                        u[-1,0,k+1] += u[-1,1,k] 
    
                        # Top left corner
                        u[0,0,k+1] += u[0,1,k]     
        return u          


         



if __name__ == '__main__':

   # Center domain is in center ! 
#       .         .      W/,L/2 
#       .         0       . 
#    -W/,-L/2     .       .  

    # Final time
    T = 0.01

    # Domain dimensions
    height = 10. #2
    width = 10. #4

    # Wave speed
    c = 343

    # Number of time steps
    nt = 400

    # Grid points in x direction
    nx = 125 # 250

    # Grid points in y direction
    ny = 125
    
    # Source term position
    xpos = 0.0
    ypos = -0.5

# microphone term position
    x_micro1 = -3.5
    y_micro1 = -1.5

    x_micro2 = -1.5
    y_micro2 = -1.5

    x_micro3 = 1.5
    y_micro3 = -1.5

    x_micro4 = 3.5
    y_micro4 = -1.5

    wave_eq = wave2d(height,width,T,nx,ny,nt,c)

    # Initial value functions
    f = lambda x,y: np.exp(-10*((x-xpos)**2+(y-ypos)**2))
    g = lambda x,y: 0 

    u = wave_eq.solve(f,g)

    x = wave_eq.x
    y = wave_eq.y

    frames = []
    #fig = plt.figure(1,(16,8))

# find node arround microphone
    xm1 =  (round(x_micro1/(width/nx)))
    ym1 =  (round(y_micro1/(height/ny)))

    xm2 =  (round(x_micro2/(width/nx)))
    ym2 =  (round(y_micro2/(height/ny)))

    xm3 =   (round(x_micro3/(width/nx)))
    ym3 =   (round(y_micro3/(height/ny)))

    xm4 =  (round(x_micro4/(width/nx)))
    ym4 =  (round(y_micro4/(height/ny)))

# if positive domain we double 
    if xm1>0 :
        xm1 +=xm1
    if xm2>0 :
        xm2 +=xm2
    if xm3>0 :
        xm3 +=xm3
    if xm4>0 :
        xm4 +=xm4
    if ym1>0 :
        ym1 +=ym1
    if ym2>0 :
        ym2 +=ym2
    if ym3>0 :
        ym3 +=ym3
    if ym4>0 :
        ym4 +=ym4

    xm1 = abs(xm1)
    ym1 = abs(ym1)

    xm2 = abs(xm2)
    ym2 = abs(ym2)

    xm3 = abs(xm3)
    ym3 = abs(ym3)

    xm4 = abs(xm4)
    ym4 = abs(ym4)

    # Setup figure and subplots
    fig = plt.figure(1,(16,8))
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)

    for k in range(nt+1):
        frame = ax1.imshow(u[:,:,k],extent=[x[0],x[-1],y[0],y[-1]])
# for view position of your microphone
        frame1, = ax1.plot(x_micro1,y_micro1, 'ro')
        frame2, = ax1.plot(x_micro2,y_micro2, 'bo')
        frame3, = ax1.plot(x_micro3,y_micro3, 'ks')
        frame4, = ax1.plot(x_micro4,y_micro4, 'gs')

        p1, = ax2.plot(k,u[xm1,ym1,k], 'ro')
        p2, = ax2.plot(k,u[xm2,ym2,k], 'bo')
        p3, = ax2.plot(k,u[xm3,ym3,k], 'ks')
        p4, = ax2.plot(k,u[xm4,ym4,k], 'gs')
        
        frames.append([frame, frame1,frame2,frame3,frame4, p1, p2, p3, p4])
       # frames.append([frame, p1, p2, p3, p4])

        plt.legend([p1, p2, p3, p4], ["1", "2", "3", "4"])
    ani = animation.ArtistAnimation(fig,frames,interval=50,
                         blit=True,repeat_delay=1000)
    ani.save('wave2d.mp4')
plt.show()

#Show final iteration and evolution of variable in captor
    # fig2 = plt.figure(1,(16,8))

    # plt.plot(k,u[xm1,ym1,k], 'ro', k,u[xm2,ym2,k], 'bo', k,u[xm3,ym3,k], 'ks', k,u[xm4,ym4,k], 'gs')
    # frames.append([frame, frame1,frame2,frame3,frame4, p1, p2, p3, p4])
    # frames.append([frame, p1, p2, p3, p4])
# plt.show()




