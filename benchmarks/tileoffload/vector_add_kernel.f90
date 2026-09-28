module tileoff_vector_add_kernel
contains

  subroutine vector_add_prepare(a, b, c)
    real :: a(:), b(:), c(:)

    !$tileoff update device(a)
    !$tileoff update device(b)
    !$tileoff update device(c)
  end subroutine

  subroutine vector_add_compute(a, b, c)
    real :: a(:), b(:), c(:)
    integer :: i

    !$tileoff parallel tile(128) pack(a:device, b:device, c:device)
    do i = 1, size(c)
      c(i) = a(i) + b(i)
    end do
  end subroutine

  subroutine vector_add_fetch(c)
    real :: c(:)

    !$tileoff update host(c)
  end subroutine

  subroutine vector_add_release(a, b, c)
    real :: a(:), b(:), c(:)

    !$tileoff release(a, b, c)
  end subroutine

end module

