module tileoff_update_before_launch_kernels
contains

  subroutine update_then_launch(a, b, c)
    real :: a(:), b(:), c(:)
    integer :: i

    !$tileoff update device(a)
    !$tileoff update device(b)

    !$tileoff parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = a(i) + b(i)
    end do

    !$tileoff update host(c)
    !$tileoff release all
  end subroutine

end module

