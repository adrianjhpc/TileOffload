module tileoff_matmul_2d_tf32_packed_kernel
contains
  subroutine packed_prepare(a,b,c,ap)
    real :: a(:,:),b(:,:),c(:,:),ap(:,:)
    !$tileoff enter data copyin(a,b) create(c,ap)
  end subroutine

  subroutine packed_pack(a,ap)
    real :: a(:,:),ap(:,:)
    integer :: i,p
    ! AP(p,i) represents A(i,p); the padded rows of AP are never read.
    !$tileoff parallel tile(32,8) pack(a:device,ap:device)
    do i=1,size(a,1)
      do p=1,size(a,2)
        ap(p,i)=a(i,p)
      enddo
    enddo
  end subroutine

  subroutine packed_compute(ap,b,c)
    real :: ap(:,:),b(:,:),c(:,:),acc
    integer :: i,j,p
    !$tileoff parallel tile(64,64,32) pack(ap:device,b:device,c:device) matmul_precision(tf32)
    do j=1,size(c,2)
      do i=1,size(c,1)
        acc=0.0
        ! Logical K comes from B, not the padded AP leading dimension.
        do p=1,size(b,1)
          acc=acc+ap(p,i)*b(p,j)
        enddo
        c(i,j)=acc
      enddo
    enddo
  end subroutine

  subroutine packed_fetch(c)
    real :: c(:,:)
    !$tileoff update host(c)
  end subroutine

  subroutine packed_release(a,b,c,ap)
    real :: a(:,:),b(:,:),c(:,:),ap(:,:)
    !$tileoff exit data delete(a,b,c,ap)
  end subroutine
end module
