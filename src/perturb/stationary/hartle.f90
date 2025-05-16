module rot_gr_mod
    ! NOT TESTED
    !
    ! Below needs verification:
    ! Q_result = Q/\Omega_s^2, where Q is defined by Q_{ij} = Q (n_i n_j - 1/3 \delta_{ij})
    use type_mod,only: wp
    implicit none
    private

    !Integration parameters
    logical::adaptive=.true.
    real(wp)::ode_eps=1.e-6_wp
    real(wp)::ode_eps_hmin=0 !Adaptive mesh method takes infinite time if gamma is not smooth; Set this value to non-zero to use adaptive
    integer::ug_steps=3000

    real(wp),allocatable,dimension(:)::t_rot
    real(wp),allocatable,dimension(:,:)::x_rot

    subroutine get_rot(moi,Q)
        !
        ! w: frame drag freq; Omega: star spin rate
        ! x(1) = w/Omega-1
        ! x(2) = dx(1)/dr
        use odeSolver_mod, only:rk45ad
        use bg_mod,only:tovs=>bgs,pt_i
        use io_mod,only:reallocate_a
        real(wp),intent(out)::moi,Q
        real(wp)::t,x(6),tb
        real(wp)::h,hmin
        real(wp)::emax,eps_hmin
        integer::isave,i
        integer::nold
        real(wp)::Rs

            emax=ode_eps
            eps_hmin=ode_eps_hmin
            isave=1 !save r position for faster interpolation with get_bg
            t=tovs(1)%r

            x = initial(t)

            i=1

            if (allocated(t_rot)) deallocate(t_rot)
            if (allocated(x_rot)) deallocate(x_rot)
            allocate(t_rot(0),x_rot(6,0))
            do
                if (size(pt_i)-i+1>0) tb=tovs(pt_i(i))%r
                if (size(pt_i)-i+1==0) tb=tovs(size(tovs))%r

                if (adaptive) then
                    h=(tb-t)/500
                    hmin=(tb-t)*eps_hmin
                    call rk45ad(eq_IQ,t,x,h,tb,emax,.true.,hmin=hmin)
                else
                    call rk4ug(eq_IQ,t,x,ug_steps,tb,.true.)
                endif

                !save solutions
                nold=size(t_rot)
                t_rot=reallocate_a(t_rot,nold+size(tp))
                x_rot=reallocate_a(x_rot,6,nold+size(tp))
                t_rot(nold+1:size(t_rot))=tp
                x_rot(:,nold+1:size(t_rot))=xp

                if (size(pt_i)-i+1==0) exit

                t=tovs(pt_i(i)+1)%r
                !All dependent variables are continuous across stellar surface
                x=x
                i=i+1
            enddo
            Rs=tovs(size(tovs))%r

            !All dependent variables are continuous across stellar surface
            x=x

!            moi= Rs**4/(6*x(1)/x(2)+2*Rs)
            moi= Rs**3/2*(x(1)-1)
            print*, 'compare mois:' , moi, Rs**4/(6*x(1)/x(2)+2*Rs)

            Q = solve_Q(Rs,x)

        contains

        function initial(t) result(res)
            use type_mod,only:G,c
            use bg_mod,only:get_tov_metric,get_bg_r
            real(wp), intent(in):: t
            real(wp),dimension(6):: res
            real(wp)::gc2,gc4,r,p,rho,m,nu,ga,A,eLam,dLam,dnu,eNu

            r=t
            gc2=G/c**2
            gc4=G/c**4

            call get_bg_r(t,p,rho,m,nu,ga,A,isave)
            call get_tov_metric(t,p,rho,m,nu,eLam,dLam,dnu,eNu)

            p=gc4*p; rho=gc2*rho !unit conversion

            res(1)= 1._wp + 8._wp/5*pi*(rho+p)*r**2
            res(2)= 16._wp/5*pi*(rho+p)*r**2
            res(3)= r**2
            res(4)= (-pi*2*(rho/3+p) -pi*4/3*(rho+p)/eNu*res(1)**2)*r**4
            res(5)= r**2
            res(6)= -pi*2*(rho/3+p)*r**4

        end function initial

        subroutine eq_IQ(r,y,f)
            use type_mod,only:G,c
            use bg_mod,only:get_tov_metric,get_bg_r
            real(wp), intent(in):: r, y(:)
            real(wp), intent(out):: f(:)
            real(wp)::gc2,gc4,r,p,rho,m,nu,ga,A,eLam,dLam,dnu,eNu

            gc2=G/c**2
            gc4=G/c**4

            call get_bg_r(r,p,rho,m,nu,ga,A,isave)
            call get_tov_metric(r,p,rho,m,nu,eLam,dLam,dnu,eNu)

            p=gc4*p; rho=gc2*rho; m=gc2*m !unit conversion

            f=w0_u2_h2_eq(r,elam,rho,p,m,y)

        end subroutine

        function solve_Q(r,y) result(res)
            use type_mod,only:G,c
            use bg_mod,only:get_tov_metric,get_bg_r
            real(wp), intent(in):: r, y(6)
            real(wp),dimension(6):: res
            real(wp)::m
            real(wp)::z,JJ,Q21,Q22
            real(wp)::const_A

            call get_bg_r(r,p,rho,m,nu,ga,A,isave)
            m = G*m/c**2 !unit conversion

            z=r/m-1
            JJ=(r**3/2*(y(1)-1))**2
            Q21=sqrt(z**2-1)*((3*z**2-2)/(z**2-1) - 1.5*z*log((z+1)/(z-1)) )
            Q22=1.5*(z**2-1)*log((z+1)/(z-1)) - (3*z**3-5*z)/(z**2-1)

            solve_const: block
                !invert a 2x2 matrix
                real(wp)::a,b,c,d,p1,p2

                a = Q22
                b = -y(5)
                c = 2*m/r/sqrt(1-2*m/r)*Q21
                d = -y(6)
                p1 = y(3) - JJ*(1._wp/m/r**3 + 1._wp/r**4)
                p2 = y(4) + JJ/r**4

                const_A = (d*p1 - b*p2)/(a*d-b*c)
            end block solve_const


            res = -(JJ/m + 8*const_A*m**3/5)
            ! Check the typo in the factor 8/5 (corrected from 16/5 in Hartle's original paper)
            ! https://ui.adsabs.harvard.edu/abs/1968ApJ...153..807H/abstract
            ! The minus sign follows Yagi and Yunes 2013
            ! https://journals.aps.org/prd/abstract/10.1103/PhysRevD.88.023009

        end function

    end subroutine get_rot

    function w0_eq(r,elam,rho,p,y) result(res)
        !Chan 2016 PRD 93 024033 Eq. (3.1)
        !Hartle 1967 Eq. (46)
        !For computing I
        use type_mod,only:pi
        real(wp),intent(in)::r,elam,rho,p
        real(wp),dimension(2),intent(in)::y
        real(wp),dimension(2)::res

        res(1) = y(2)
        res(2) = pi*16*(rho+p)*elam*y(1) - (4./r - pi*4*(rho+p)*elam )*y(2)

    end function w0_eq

    function w0_u2_h2_eq(r,elam,eNu,rho,p,m,y) result(res)
        !w0 equations:
        !   Chan 2016 PRD 93 024033 Eq. (3.1)
        !   Hartle 1967 Eq. (46)
        !u2, h2 equations:
        !Hartle 1967 Eq. (125-126)
        ! u2 is used to replace nu = h2 + k2 in Hartle due to conflict of symbol
        use type_mod,only:pi
        real(wp),intent(in)::r,elam,eNu,rho,p,m
        real(wp),dimension(6),intent(in)::y
        real(wp),dimension(6)::res

        real(wp)::w,dw,u2_p,h2_p,u2_h,h2_h
        real(wp)::jj,jjp,dnu

        w=y(1)
        dw=y(2)
        h2_p=y(3)
        u2_p=y(4)
        h2_h=y(5)
        u2_h=y(6)

        dnu = (m/r**2+pi*8*r*p)*elam
        jj = 1._wp/sqrt(elam*eNu)
        jjp=-jj*pi*8*r*(rho+p)*elam

        res(1) = dw
        res(2) = pi*16*(rho+p)*elam*w - (4./r - pi*4*(rho+p)*elam )*dw
        res(3) = -4._wp/r**2*elam/dnu*u2_p + (-dnu + elam/dnu*(pi*8*(rho+p)-4*m/r**3))*h2_p &
        + r**3/6*(r*dnu/2-elam/r/dnu)*jj*dw**2 - r**2/3*(r*dnu/2+ elam/r/dnu)*jjp*w**2
        res(4) = -dnu*h2_p +(1._wp/r+dnu/2)*(-r**3/3*jjp*w**2+jj/3*r**4*dw**2)
        res(5) = -4._wp/r**2*elam/dnu*u2_h + (-dnu + elam/dnu*(pi*8*(rho+p)-4*m/r**3))*h2_h
        res(6) = -dnu*h2_h

    end function w0_u2_h2_eq

end module rot_gr_mod
