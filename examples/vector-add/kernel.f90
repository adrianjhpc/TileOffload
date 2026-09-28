subroutine compute_add(n, a, b, c)
  integer :: n
  real :: a(n), b(n), c(n)
  integer :: i

  !$tileoff parallel tile(128) pack(a:device, b:device, c:device)
  do i = 1, n
    c(i) = a(i) + b(i)
  end do

  !$tileoff update host(c)
  !$tileoff release all
end subroutine
