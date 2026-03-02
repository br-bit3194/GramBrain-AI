'use client'

import Link from 'next/link'
import { useAppStore } from '@/store/appStore'
import { FiMenu, FiX, FiLogOut } from 'react-icons/fi'
import { useState } from 'react'

export function Header() {
  const { user, clearStore } = useAppStore()
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const handleLogout = () => {
    clearStore()
    setIsMenuOpen(false)
  }

  return (
    <header className="bg-gradient-to-r from-green-700 via-green-600 to-emerald-600 shadow-lg sticky top-0 z-50">
      <nav className="container-custom flex items-center justify-between h-16">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 bg-yellow-400 rounded-lg flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
            <span className="text-green-700 font-bold text-lg">🌾</span>
          </div>
          <span className="font-bold text-lg text-white hidden sm:inline group-hover:text-yellow-100 transition">
            GramBrain AI
          </span>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-8">
          {user ? (
            <>
              <Link href="/dashboard" className="text-white hover:text-yellow-200 transition font-medium">
                Dashboard
              </Link>
              <Link href="/query" className="text-white hover:text-yellow-200 transition font-medium">
                Query
              </Link>
              <Link href="/farms" className="text-white hover:text-yellow-200 transition font-medium">
                Farms
              </Link>
              <Link href="/marketplace" className="text-white hover:text-yellow-200 transition font-medium">
                Marketplace
              </Link>
              <div className="flex items-center gap-4 pl-8 border-l border-green-500">
                <span className="text-sm text-yellow-100">{user.name}</span>
                <button
                  onClick={handleLogout}
                  className="text-white hover:text-red-200 transition font-medium flex items-center gap-2"
                  title="Logout"
                >
                  <FiLogOut size={20} />
                  Logout
                </button>
              </div>
            </>
          ) : (
            <>
              <Link href="/login" className="text-white hover:text-yellow-200 transition font-medium">
                Login
              </Link>
              <Link href="/register" className="bg-yellow-400 text-green-700 px-6 py-2 rounded-lg font-semibold hover:bg-yellow-300 transition shadow-md">
                Register
              </Link>
            </>
          )}
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-white hover:text-yellow-200 transition"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label="Toggle menu"
        >
          {isMenuOpen ? <FiX size={24} /> : <FiMenu size={24} />}
        </button>
      </nav>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="md:hidden bg-green-700 border-t border-green-600">
          <div className="container-custom py-4 flex flex-col gap-4">
            {user ? (
              <>
                <Link href="/dashboard" className="text-white hover:text-yellow-200 transition font-medium">
                  Dashboard
                </Link>
                <Link href="/query" className="text-white hover:text-yellow-200 transition font-medium">
                  Query
                </Link>
                <Link href="/farms" className="text-white hover:text-yellow-200 transition font-medium">
                  Farms
                </Link>
                <Link href="/marketplace" className="text-white hover:text-yellow-200 transition font-medium">
                  Marketplace
                </Link>
                <button
                  onClick={handleLogout}
                  className="text-left text-white hover:text-red-200 transition font-medium flex items-center gap-2"
                >
                  <FiLogOut size={20} />
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link href="/login" className="text-white hover:text-yellow-200 transition font-medium">
                  Login
                </Link>
                <Link href="/register" className="bg-yellow-400 text-green-700 px-6 py-2 rounded-lg font-semibold hover:bg-yellow-300 transition text-center">
                  Register
                </Link>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  )
}
