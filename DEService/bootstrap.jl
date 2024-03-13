(pwd() != @__DIR__) && cd(@__DIR__) # allow starting app from bin/ dir

using DEService
const UserApp = DEService
DEService.main()
