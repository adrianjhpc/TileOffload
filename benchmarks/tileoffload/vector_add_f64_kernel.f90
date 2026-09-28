module tileoff_vector_add_f64_kernel
contains

  subroutine vector_add_f64_prepare(a, b, c)
    real(8) :: a(:), b(:), c(:)

    !$tileoff enter data copyin(a, b) create(c)
  end subroutine

  subroutine vector_add_f64_compute(a, b, c)
    real(8) :: a(:), b(:), c(:)
    integer :: i

    !$tileoff parallel tile(128)
    do i = 1, size(c)
      c(i) = a(i) + b(i)
    end do
  end subroutine

  subroutine vector_add_f64_fetch(c)
    real(8) :: c(:)

    !$tileoff update host(c)
  end subroutine

  subroutine vector_add_f64_release(a, b, c)
    real(8) :: a(:), b(:), c(:)

    !$tileoff exit data delete(a, b, c)
  end subroutine

end module

