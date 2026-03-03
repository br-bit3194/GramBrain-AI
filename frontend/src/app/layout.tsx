'use client'

import { useEffect } from 'react'
import { Header } from '@/components/layout/Header'
import { Footer } from '@/components/layout/Footer'
import { initializeApiClient } from '@/lib/apiInit'
import '@/styles/globals.css'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Initialize API client with token getter on mount
  useEffect(() => {
    initializeApiClient()
  }, [])

  return (
    <html lang="en">
      <body suppressHydrationWarning>
        <div className="flex flex-col min-h-screen">
          <Header />
          <main className="flex-grow">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  )
}
