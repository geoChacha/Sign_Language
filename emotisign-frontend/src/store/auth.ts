import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import Cookies from 'js-cookie';
import { User } from '@/types';
import api from '@/lib/api';

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isGuest: boolean;
  setUser: (user: User | null) => void;
  login: (username: string, password: string) => Promise<void>;
  register: (data: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
    role: 'hearing' | 'deaf' | 'admin';
    preferred_sign_language: 'ASL' | 'PSL';
  }) => Promise<void>;
  logout: () => Promise<void>;
  fetchUser: () => Promise<void>;
  continueAsGuest: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isLoading: false,
      isAuthenticated: false,
      isGuest: false,

      setUser: (user) => set({ user, isAuthenticated: !!user }),

      login: async (username, password) => {
        set({ isLoading: true });
        try {
          const response = await api.login(username, password);
          set({
            user: response.user,
            isAuthenticated: true,
            isGuest: false,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      register: async (data) => {
        set({ isLoading: true });
        try {
          await api.register(data);
          set({ isLoading: false });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },

      logout: async () => {
        set({ isLoading: true });
        try {
          await api.logout();
        } finally {
          Cookies.remove('access_token');
          set({
            user: null,
            isAuthenticated: false,
            isGuest: false,
            isLoading: false,
          });
        }
      },

      fetchUser: async () => {
        const token = Cookies.get('access_token');
        if (!token) {
          set({ isAuthenticated: false, user: null });
          return;
        }

        set({ isLoading: true });
        try {
          const user = await api.getCurrentUser();
          set({ user, isAuthenticated: true, isLoading: false });
        } catch {
          Cookies.remove('access_token');
          set({ user: null, isAuthenticated: false, isLoading: false });
        }
      },

      continueAsGuest: () => {
        set({ isGuest: true, isAuthenticated: false, user: null });
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ isGuest: state.isGuest }),
    }
  )
);
