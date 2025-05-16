module love_gr_mod
    implicit none
    private
    public::get_love_gr_fluid
    public::love_eq_ell2
    public::t_lo,x_lo
    public::ug_steps
    public::adaptive

    integer,parameter::dp=kind(1.d0)
    integer,parameter::wp=dp

    !Integration parameters
    logical::adaptive=.true.
    real(wp)::ode_eps=1.e-6_wp
    real(wp)::ode_eps_hmin=0 !Adaptive mesh method takes infinite time if gamma is not smooth; Set this value to non-zero to use adaptive
    integer::ug_steps=3000

    real(wp),allocatable,dimension(:)::t_lo
    real(wp),allocatable,dimension(:)::x_lo

contains

    subroutine get_love_gr_fluid(ell,kl)
        !t: independent variable in general. In this problem it is the distance from stellar center
        !x: dependent variable in general. In this problem it is (r H_0^{\prime}/H_0)
        !Original equations Eq. (15),(23) of Hinderer Astrophy. J. 677 1216 (2008) [!See the erratum of Eq. (23)] https://iopscience.iop.org/article/10.1086/533487
        !Rewritten as Eq. (4.7) T K Chan PRD 044017 (2015) https://arxiv.org/abs/1411.7141?context=gr-qc
        use type_mod,only:pi4,c
        use odeSolver_mod
!        use bg_mod,only:bgs,pt_i
        use bg_mod,only:tovs=>bgs,pt_i
        use io_mod,only:reallocate_a
        real(wp),intent(in)::ell
        real(wp),intent(out)::kl
        real(wp)::t,x(1),tb
        real(wp)::h,hmin
        real(wp)::m_i,p_i,drho_i
        real(wp)::emax,eps_hmin
        integer::isave,i
        integer::nold

            emax=ode_eps
            eps_hmin=ode_eps_hmin
            isave=1 !save r position for faster interpolation with get_bg
            t=tovs(1)%r
            x(1)=ell !initial condition
            i=1

            if (allocated(t_lo)) deallocate(t_lo)
            if (allocated(x_lo)) deallocate(x_lo)
            allocate(t_lo(0),x_lo(0))
            do
                if (size(pt_i)-i+1>0) tb=tovs(pt_i(i))%r
                if (size(pt_i)-i+1==0) tb=tovs(size(tovs))%r

                if (adaptive) then
                    h=(tb-t)/500
                    hmin=(tb-t)*eps_hmin
                    call rk45ad(eq_chan15,t,x,h,tb,emax,.true.,hmin=hmin)
                else
                    call rk4ug(eq_chan15,t,x,ug_steps,tb,.true.)
                endif

                !save solutions
                nold=size(t_lo)
                t_lo=reallocate_a(t_lo,nold+size(tp))
                x_lo=reallocate_a(x_lo,nold+size(tp))
                t_lo(nold+1:size(t_lo))=tp
                x_lo(nold+1:size(t_lo))=xp(1,:)

                if (size(pt_i)-i+1==0) exit
                t=tovs(pt_i(i)+1)%r
                m_i=tovs(pt_i(i)+1)%m
                p_i=tovs(pt_i(i)+1)%p
                drho_i=tovs(pt_i(i)+1)%rho-tovs(pt_i(i))%rho
                 !Junction condition
                ! Check out https://doi.org/10.1103/PhysRevD.102.028501
                x=x + pi4*t**3/(m_i+pi4*t**3*p_i/c**2)*drho_i
                i=i+1
            enddo
            t=tovs(size(tovs))%r
            !Junction condition between surface and vacuum
            ! Check out https://doi.org/10.1103/PhysRevD.102.028501
            x=x-pi4*t**3/(tovs(size(tovs))%m)*tovs(size(tovs))%rho

            if (nint(ell)/=2) then !Future work: expand this part to include other ell
                write(*,*) 'err: get_love_gr_fluid ell /= 2'
                stop
            end if
            kl=love_eq_ell2(tovs(size(tovs))%m,tovs(size(tovs))%r,x(1))

        contains

        subroutine eq_chan15(t,x,f) !T K Chan PRD 044017 (2015)
            use type_mod,only:G,c
            use bg_mod,only:get_tov_metric,get_bg_r
            real(wp), intent(in):: t, x(:)
            real(wp), intent(out):: f(:)
            real(wp)::gc2,gc4,r,p,rho,m,nu,ga,A,eLam,dLam,dnu,eNu
            real(wp)::cs2

            r=t
            gc2=G/c**2
            gc4=G/c**4

            call get_bg_r(t,p,rho,m,nu,ga,A,isave)
            call get_tov_metric(t,p,rho,m,nu,eLam,dLam,dnu,eNu)

            p=gc4*p; rho=gc2*rho; m=gc2*m !unit conversion
            cs2=ga*p/(rho+p)

            f(1)=dy_chan15(ell,r,eLam,dnu,rho,P,cs2,x(1))
        end subroutine

    end subroutine get_love_gr_fluid

    function dy_chan15(ell,r,eLam,dnu,rho,P,cs2,y)
        use type_mod,only:pi4
        real(wp),intent(in)::ell,r,eLam,dnu,rho,P,cs2,y
        real(wp)::Q
        real(wp)::dy_chan15
        Q=pi4*eLam*(5._wp*rho+9._wp*p+(rho+p)/cs2)-ell*(ell+1)*eLam/r**2-dnu**2
        dy_chan15=-(y**2+y*eLam*(1._wp+pi4*r**2*(p-rho))+r**2*Q)/r
    end function dy_chan15





    !Calculate the Love number from the vacuum solution
    function love_eq_ell2(m,r,y)
        use type_mod,only:G,c
        real(wp),intent(in)::m,r,y
        real(wp)::Co
        real(wp)::t1,t2,t3
        real(wp)::love_eq_ell2

        Co = G/c**2*m/r
        t1 = 8._wp/5._wp*Co**5*(1._wp-2._wp*Co)**2*(2._wp*Co*(y-1._wp)+2._wp-y)
        t2 = 2._wp*Co*(4._wp*(y+1._wp)*Co**4+(6._wp*y-4._wp)*Co**3+(26._wp-22._wp*y)*Co**2 &
        +3._wp*Co*(5._wp*y-8._wp)-3._wp*y+6._wp)
        t3 = 3._wp*(1._wp-2._wp*Co)**2*(2._wp*Co*(y-1._wp)-y+2._wp)*log(1._wp-2._wp*Co)

        love_eq_ell2 = t1/(t2+t3)

        ! Low Compactness Expansion
        ! Samson T K Chan (2016)
        If (Co<=3.e-3_wp) then
            t1 = (2._wp - y) / 2._wp / (y + 3._wp)
            t2 = 5._wp * Co * (y**2 + 2._wp*y - 6._wp)/2._wp/(y + 3._wp)**2
            t3 = 5._wp*Co**2 * (11._wp*y**3 + 66._wp*y**2 + 52._wp*y - 204._wp)/14._wp/(y + 3._wp)**3
            love_eq_ell2 = t1 + t2 + t3
        endif
    end function love_eq_ell2
end module

