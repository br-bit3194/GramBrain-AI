'use client'

import Link from 'next/link'

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-gradient-to-b from-green-900 to-green-950 text-green-100 mt-16">
      <div className="container-custom py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* About */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl">🌾</span>
              <h3 className="text-white font-bold text-lg">GramBrain AI</h3>
            </div>
            <p className="text-sm text-green-200">
              Multi-agent AI platform empowering farmers with intelligent, real-time agricultural insights and recommendations.
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
              <span className="text-yellow-400">🌱</span> Product
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/dashboard" className="text-green-200 hover:text-yellow-300 transition">
                  Dashboard
                </Link>
              </li>
              <li>
                <Link href="/query" className="text-green-200 hover:text-yellow-300 transition">
                  AI Query
                </Link>
              </li>
              <li>
                <Link href="/farms" className="text-green-200 hover:text-yellow-300 transition">
                  Farm Management
                </Link>
              </li>
              <li>
                <Link href="/marketplace" className="text-green-200 hover:text-yellow-300 transition">
                  Marketplace
                </Link>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
              <span className="text-yellow-400">🏢</span> Company
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  About Us
                </Link>
              </li>
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  Blog
                </Link>
              </li>
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  Contact
                </Link>
              </li>
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  Careers
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
              <span className="text-yellow-400">⚖️</span> Legal
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  License
                </Link>
              </li>
              <li>
                <Link href="/" className="text-green-200 hover:text-yellow-300 transition">
                  Disclaimer
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-green-800 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <p className="text-sm text-green-200">
              © {currentYear} GramBrain AI. All rights reserved. 🌾
            </p>
            <div className="flex gap-6 text-sm">
              <a href="https://github.com" className="text-green-200 hover:text-yellow-300 transition flex items-center gap-2">
                <span>💻</span> GitHub
              </a>
              <a href="https://twitter.com" className="text-green-200 hover:text-yellow-300 transition flex items-center gap-2">
                <span>🐦</span> Twitter
              </a>
              <a href="https://linkedin.com" className="text-green-200 hover:text-yellow-300 transition flex items-center gap-2">
                <span>💼</span> LinkedIn
              </a>
            </div>
          </div>
        </div>

        {/* Bottom tagline */}
        <div className="mt-8 pt-8 border-t border-green-800 text-center">
          <p className="text-sm text-green-300 italic">
            "Empowering farmers with AI-driven intelligence for sustainable and profitable agriculture"
          </p>
        </div>
      </div>
    </footer>
  )
}
